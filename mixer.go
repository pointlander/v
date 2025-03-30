// Copyright 2025 The v Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

const (
	// Size is the number of histograms
	Size = 8
	// Order is the order of the markov model
	Order = 7
)

const (
	// CDF16Fixed is the shift for 16 bit coders
	CDF16Fixed = 16 - 3
	// CDF16Scale is the scale for 16 bit coder
	CDF16Scale = 1 << CDF16Fixed
	// CDF16Rate is the damping factor for 16 bit coder
	CDF16Rate = 5
)

// Mix is a mixer
type Mix interface {
	Copy() Mix
	Add(byte)
	Mix() [InputSize]float32
}

// Mix is a mixer
type CrossMix interface {
	Copy() CrossMix
	Add(byte, byte)
	Mix() [InputSize]float32
}

type CDF16 struct {
	Size   int
	Rate   int
	Model  []uint16
	Mixin  [][]uint16
	Verify bool
}

type Filtered16 interface {
	GetModel() []uint16
	Copy() Filtered16
	Update(s uint16)
}

type CDF16Maker func(size, rate int) Filtered16

func NewCDF16(verify bool) CDF16Maker {
	return func(size, rate int) Filtered16 {
		if size != 256 {
			panic("size is not 256")
		}
		model, sum := make([]uint16, size+1), 0
		for i := range model {
			model[i] = uint16(sum)
			sum += 32
		}

		mixin := make([][]uint16, size)

		for i := range mixin {
			sum, m := 0, make([]uint16, size+1)
			for j := range m {
				m[j] = uint16(sum)
				sum++
				if j == i {
					sum += CDF16Scale - size
				}
			}
			mixin[i] = m
		}

		return &CDF16{
			Size:   size,
			Rate:   rate,
			Model:  model,
			Mixin:  mixin,
			Verify: verify,
		}
	}
}

// Copy copies the model
func (c *CDF16) Copy() Filtered16 {
	model := make([]uint16, len(c.Model))
	copy(model, c.Model)
	return &CDF16{
		Size:   c.Size,
		Rate:   c.Rate,
		Model:  model,
		Mixin:  c.Mixin,
		Verify: c.Verify,
	}
}

// GetModel gets the cdf
func (c *CDF16) GetModel() []uint16 {
	return c.Model
}

// Update the cdf
func (c *CDF16) Update(s uint16) {
	model, mixin := c.Model, c.Mixin[s]
	size, rate := len(model)-1, c.Rate

	if c.Verify {
		for i := 1; i < size; i++ {
			a, b := int(model[i]), int(mixin[i])
			if a < 0 {
				panic("a is less than zero")
			}
			if b < 0 {
				panic("b is less than zero")
			}
			model[i] = uint16(a + ((b - a) >> rate))
		}
		if model[size] != CDF16Scale {
			panic("cdf scale is incorrect")
		}
		for i := 1; i < len(model); i++ {
			if a, b := model[i], model[i-1]; a < b {
				panic(fmt.Sprintf("invalid cdf %v,%v < %v,%v", i, a, i-1, b))
			} else if a == b {
				panic(fmt.Sprintf("invalid cdf %v,%v = %v,%v", i, a, i-1, b))
			}
		}
	} else {
		for i := 1; i < size; i++ {
			a, b := int(model[i]), int(mixin[i])
			model[i] = uint16(a + ((b - a) >> rate))
		}
	}
}

// Markov is a markov model
type Markov [Order + 1]byte

// Histogram is a buffered histogram
type Histogram struct {
	Vector [256]byte
	Buffer [128]byte
	Index  int
	Size   int
}

// NewHistogram make a new histogram
func NewHistogram(size int) Histogram {
	h := Histogram{
		Size: size,
	}
	return h
}

// Add adds a symbol to the histogram
func (h *Histogram) Add(s byte) {
	index := (h.Index + 1) % h.Size
	if symbol := h.Buffer[index]; h.Vector[symbol] > 0 {
		h.Vector[symbol]--
	}
	h.Buffer[index] = s
	h.Vector[s]++
	h.Index = index
}

// Filtered is a filtered counter
type Filtered struct {
	Markov  Markov
	Filters []Filtered16
}

// NewFiltered makes a new filtered counter
func NewFiltered() *Filtered {
	cdf := NewCDF16(false)
	filters := make([]Filtered16, Size)
	for i := range filters {
		filters[i] = cdf(256, i+1)
	}
	return &Filtered{
		Filters: filters,
	}
}

// Copy copies the filter
func (f Filtered) Copy() Mix {
	filters := make([]Filtered16, len(f.Filters))
	for i := range filters {
		filters[i] = f.Filters[i].Copy()
	}
	return &Filtered{
		Markov:  f.Markov,
		Filters: filters,
	}
}

// Add adds a symbol to a filter
func (f Filtered) Add(s byte) {
	for i := range f.Filters {
		f.Filters[i].Update(uint16(s))
	}
	for k := Order; k > 0; k-- {
		f.Markov[k] = f.Markov[k-1]
	}
	f.Markov[0] = s
}

// Mix mixes the filters outputting a matrix
func (f Filtered) Mix() [InputSize]float32 {
	x := NewMatrix(256, Size+Order+1)
	for i := range f.Filters {
		model := f.Filters[i].GetModel()
		last, sum := uint16(0), float32(0.0)
		for _, v := range model[1:] {
			sum += float32(v - last)
			last = v
		}
		last = 0
		for _, v := range model[1:] {
			x.Data = append(x.Data, float32(v-last)/sum)
			last = v
		}
	}
	for _, v := range f.Markov {
		d := make([]float32, 256)
		d[v] = 1
		x.Data = append(x.Data, d...)
	}
	return SelfAttention(x)
}

// Filtered is a filtered counter
type CrossFiltered struct {
	Markov  [2]Markov
	Filters [2][]Filtered16
}

// NewCrossFiltered makes a new cross filtered counter
func NewCrossFiltered() *CrossFiltered {
	cdf := NewCDF16(false)
	filters := [2][]Filtered16{}
	for i := range filters {
		filters[i] = make([]Filtered16, Size)
		for j := range filters[i] {
			filters[i][j] = cdf(256, i+1)
		}
	}
	return &CrossFiltered{
		Filters: filters,
	}
}

// Copy copies the filter
func (f CrossFiltered) Copy() CrossMix {
	filters := [2][]Filtered16{}
	for i := range filters {
		filters[i] = make([]Filtered16, len(f.Filters))
		for j := range filters[i] {
			filters[i][j] = f.Filters[i][j].Copy()
		}
	}
	return &CrossFiltered{
		Markov:  f.Markov,
		Filters: filters,
	}
}

// Add adds a symbol to a filter
func (f CrossFiltered) Add(s1, s2 byte) {
	for i := range f.Filters[0] {
		f.Filters[0][i].Update(uint16(s1))
	}
	for k := Order; k > 0; k-- {
		f.Markov[0][k] = f.Markov[0][k-1]
	}
	f.Markov[0][0] = s1
	for i := range f.Filters[1] {
		f.Filters[1][i].Update(uint16(s2))
	}
	for k := Order; k > 0; k-- {
		f.Markov[1][k] = f.Markov[1][k-1]
	}
	f.Markov[1][0] = s2
}

// Mix mixes the filters outputting a matrix
func (f CrossFiltered) Mix() [InputSize]float32 {
	x := [2]Matrix{NewMatrix(256, Size+Order+1), NewMatrix(256, Size+Order+1)}
	for i := range x {
		for j := range f.Filters[i] {
			model := f.Filters[i][j].GetModel()
			last, sum := uint16(0), float32(0.0)
			for _, v := range model[1:] {
				sum += float32(v - last)
				last = v
			}
			last = 0
			for _, v := range model[1:] {
				x[i].Data = append(x[i].Data, float32(v-last)/sum)
				last = v
			}
		}
		for _, v := range f.Markov[i] {
			d := make([]float32, 256)
			d[v] = 1
			x[i].Data = append(x[i].Data, d...)
		}
	}
	return CrossSelfAttention(x[0], x[1])
}

// Mixer mixes several histograms together
type Mixer struct {
	Markov     Markov
	Histograms []Histogram
}

// NewMixer makes a new mixer
func NewMixer() *Mixer {
	histograms := make([]Histogram, Size)
	histograms[0] = NewHistogram(1)
	histograms[1] = NewHistogram(2)
	histograms[2] = NewHistogram(4)
	histograms[3] = NewHistogram(8)
	histograms[4] = NewHistogram(16)
	histograms[5] = NewHistogram(32)
	histograms[6] = NewHistogram(64)
	histograms[7] = NewHistogram(128)
	return &Mixer{
		Histograms: histograms,
	}
}

func (m Mixer) Copy() Mix {
	histograms := make([]Histogram, Size)
	for i := range m.Histograms {
		histograms[i] = m.Histograms[i]
	}
	return &Mixer{
		Markov:     m.Markov,
		Histograms: histograms,
	}
}

// Add adds a symbol to a mixer
func (m *Mixer) Add(s byte) {
	for i := range m.Histograms {
		m.Histograms[i].Add(s)
	}
	for k := Order; k > 0; k-- {
		m.Markov[k] = m.Markov[k-1]
	}
	m.Markov[0] = s
}

// Mix mixes the histograms outputting a matrix
func (m Mixer) Mix() [InputSize]float32 {
	x := NewMatrix(256, Size+Order+1)
	for i := range m.Histograms {
		sum := float32(0.0)
		for _, v := range m.Histograms[i].Vector {
			sum += float32(v)
		}
		for _, v := range m.Histograms[i].Vector {
			x.Data = append(x.Data, float32(v)/sum)
		}
	}
	for _, v := range m.Markov {
		d := make([]float32, 256)
		d[v] = 1
		x.Data = append(x.Data, d...)
	}
	return SelfAttention(x)
}

// CrossMixer mixes several histograms together
type CrossMixer struct {
	Markov     [2]Markov
	Histograms [2][]Histogram
}

// NewCrossMixer makes a new cross mixer
func NewCrossMixer() *CrossMixer {
	histograms := [2][]Histogram{}
	for i := range histograms {
		histograms[i] = make([]Histogram, Size)
		histograms[i][0] = NewHistogram(1)
		histograms[i][1] = NewHistogram(2)
		histograms[i][2] = NewHistogram(4)
		histograms[i][3] = NewHistogram(8)
		histograms[i][4] = NewHistogram(16)
		histograms[i][5] = NewHistogram(32)
		histograms[i][6] = NewHistogram(64)
		histograms[i][7] = NewHistogram(128)
	}
	return &CrossMixer{
		Histograms: histograms,
	}
}

// Copy copies a cross mixer
func (m CrossMixer) Copy() CrossMix {
	histograms := [2][]Histogram{}
	for i := range histograms {
		histograms[i] = make([]Histogram, Size)
		for j := range m.Histograms[i] {
			histograms[i][j] = m.Histograms[i][j]
		}
	}
	return &CrossMixer{
		Markov:     m.Markov,
		Histograms: histograms,
	}
}

// Add adds a symbol to a cross mixer
func (m *CrossMixer) Add(s1, s2 byte) {
	for i := range m.Histograms[0] {
		m.Histograms[0][i].Add(s1)
	}
	for k := Order; k > 0; k-- {
		m.Markov[0][k] = m.Markov[0][k-1]
	}
	m.Markov[0][0] = s1
	for i := range m.Histograms[1] {
		m.Histograms[1][i].Add(s2)
	}
	for k := Order; k > 0; k-- {
		m.Markov[1][k] = m.Markov[1][k-1]
	}
	m.Markov[1][0] = s2
}

// Mix mixes the histograms outputting a matrix
func (m CrossMixer) Mix() [InputSize]float32 {
	x := [2]Matrix{NewMatrix(256, Size+Order+1), NewMatrix(256, Size+Order+1)}
	for i := range x {
		for j := range m.Histograms[i] {
			sum := float32(0.0)
			for _, v := range m.Histograms[i][j].Vector {
				sum += float32(v)
			}
			for _, v := range m.Histograms[i][j].Vector {
				x[i].Data = append(x[i].Data, float32(v)/sum)
			}
		}
		for _, v := range m.Markov[i] {
			d := make([]float32, 256)
			d[v] = 1
			x[i].Data = append(x[i].Data, d...)
		}
	}
	return CrossSelfAttention(x[0], x[1])
}
