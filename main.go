// Copyright 2025 The Shakespeare Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"context"
	"embed"
	"encoding/json"
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

func main() {
	rng := rand.New(rand.NewSource(1))
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
	var transforms [256][256]float32
	for i := range transforms {
		for j := range transforms[i] {
			transforms[i][j] = rng.Float32()
		}
		a := vector.Dot(transforms[i][:], transforms[i][:])
		for j := range transforms[i] {
			transforms[i][j] /= a
		}
	}
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
