// Copyright 2025 The Shakespeare Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	// InputSize is the size of the input
	InputSize = 256
)

// V is the v application
type V struct {
	Address  string `json:"address"`
	Username string `json:"username"`
	Password string `json:"password"`
}

const (
	msgFmt = "==== %s ====\n"
)

func main() {
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
