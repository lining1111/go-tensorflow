package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math/rand"
)

//三维点
func main() {
	//使用gonum生成假数据，100个点
	x_data := mat.NewDense(100, 2, nil)
	for i := 0; i < 100; i++ {
		for j := 0; j < 2; j++ {
			x_data.Set(i, j, rand.NormFloat64())
		}
	}
	// Create a matrix formatting value with a prefix using Python format...
	fx := mat.Formatted(x_data, mat.Prefix("    "), mat.FormatPython())
	// and then print with and without layout formatting.
	fmt.Printf("layout syntax:\nx = %#v\n\n", fx)

	//y= [0.1 0.2]*x + 0.3
	y_data := mat.NewDense(100, 1, nil)

	W := mat.NewDense(1, 2, []float64{0.1, 0.2})
	y_data.Mul(x_data, W.T())
	addScalar := func(_, _ int, v float64) float64 {
		return v + 0.3
	}
	y_data.Apply(addScalar, y_data)

	// Create a matrix formatting value with a prefix using Python format...
	fy := mat.Formatted(y_data, mat.Prefix("    "), mat.FormatPython())
	// and then print with and without layout formatting.
	fmt.Printf("layout syntax:\ny = %#v\n\n", fy)

	//构建一个线性模型

}
