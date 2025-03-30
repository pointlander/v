// Copyright 2025 The Shakespeare Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"context"
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"math/rand"
	"os"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/pointlander/v/vector"
)

const (
	// InputSize is the size of the input
	InputSize = 256
)

//go:embed books/*
var Data embed.FS

// V is the v application
type V struct {
	Address  string `json:"address"`
	Username string `json:"username"`
	Password string `json:"password"`
}

const (
	msgFmt = "==== %s ====\n"
)

func VDB() {
	config, err := os.Open("v.json")
	if err != nil {
		panic(err)
	}
	defer config.Close()
	data, err := ioutil.ReadAll(config)
	if err != nil {
		panic(err)
	}
	var v V
	err = json.Unmarshal(data, &v)
	if err != nil {
		panic(err)
	}

	ctx := context.Background()

	fmt.Printf(msgFmt, "start connecting to Milvus")
	c, err := client.NewClient(ctx, client.Config{
		Address:  v.Address,
		Username: v.Username,
		Password: v.Password,
	})
	if err != nil {
		panic(fmt.Errorf("failed to connect to milvus, err: %v", err.Error()))
	}
	defer c.Close()

	collectionName := `vectors`

	has, err := c.HasCollection(ctx, collectionName)
	if err != nil {
		panic(fmt.Errorf("failed to check whether collection exists: %v", err.Error()))
	}
	if has {
		_ = c.DropCollection(ctx, collectionName)
	}

	schema := entity.NewSchema().WithName(collectionName).WithDescription("vector to symbol collection").
		WithField(entity.NewField().WithName("ID").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName("Symbol").WithDataType(entity.FieldTypeVarChar).WithMaxLength(1)).
		WithField(entity.NewField().WithName("Vector").WithDataType(entity.FieldTypeFloatVector).WithDim(256))

	err = c.CreateCollection(ctx, schema, entity.DefaultShardNumber) // only 1 shard
	if err != nil {
		panic(fmt.Errorf("failed to create collection: %v", err.Error()))
	}
}

// GetTransforms generates the vector transforms
func GetTransforms() (transforms [512][256]float32) {
	rng := rand.New(rand.NewSource(1))
	for i := range transforms {
		for j := range transforms[i] {
			transforms[i][j] = rng.Float32()
		}
		a := sqrt(vector.Dot(transforms[i][:], transforms[i][:]))
		for j := range transforms[i] {
			transforms[i][j] /= a
		}
	}
	return transforms
}

var (
	// FlagInfer is the inference mode
	FlagInfer = flag.String("infer", "", "inference mode")
)

func main() {
	flag.Parse()

	if *FlagInfer != "" {
		rng := rand.New(rand.NewSource(1))
		db, err := os.Open(*FlagInfer)
		if err != nil {
			panic(err)
		}
		defer db.Close()
		transforms := GetTransforms()
		m, input := NewFiltered(), "What is love?"
		for _, v := range []byte(input) {
			m.Add(v)
		}
		for i := 0; i < 128; i++ {
			var indexes [len(transforms)]int64
			vv := m.Mix()
			for i := range transforms {
				indexes[i] = int64(math.Float32bits(2*float32(i) + vector.Dot(vv[:], transforms[i][:])))
			}
			var histogram [256]uint
			found := false
			for i := 1; i < 1024 && !found; i *= 2 {
				for j := range indexes {
					begin, end := indexes[j]-int64(i), indexes[j]+int64(i)
					if begin < 0 {
						begin = 0
					}
					if end > math.MaxUint32 {
						end = math.MaxUint32
					}
					_, err := db.Seek(begin, 0)
					if err != nil {
						panic(err)
					}
					buffer := make([]byte, end-begin+1)
					_, err = db.Read(buffer)
					if err != nil {
						panic(err)
					}
					for _, v := range buffer {
						if v != 0 {
							found = true
							histogram[v]++
						}
					}
				}
			}
			sum := uint(0)
			for _, v := range histogram {
				sum += v
			}
			total, selected, index := 0.0, rng.Float64(), 0
			for i, v := range histogram {
				total += float64(v) / float64(sum)
				if selected < total {
					index = i
					break
				}
			}
			fmt.Printf("%c", byte(index))
			m.Add(byte(index))
		}
		return
	}

	var test [4 * 1024 * 1024 * 1024]byte
	file, err := Data.Open("books/100.txt.utf-8.bz2")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	reader := bzip2.NewReader(file)
	data, err := io.ReadAll(reader)
	if err != nil {
		panic(err)
	}
	m := NewFiltered()
	m.Add(0)
	transforms := GetTransforms()
	for j, v := range data {
		vv := m.Mix()
		for i := range transforms {
			x := math.Float32bits(2*float32(i) + vector.Dot(vv[:], transforms[i][:]))
			test[x] = v
		}
		m.Add(v)
		fmt.Println(float64(j) / float64(len(data)))
	}
	out, err := os.Create("model.bin")
	if err != nil {
		panic(err)
	}
	defer out.Close()
	_, err = out.Write(test[:])
	if err != nil {
		panic(err)
	}
}
