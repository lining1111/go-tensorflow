package main

import (
	"fmt"
	tf "github.com/wamuir/graft/tensorflow"
	"github.com/wamuir/graft/tensorflow/op"
	"math/rand"
	"os"
)

func main() {
	var x_data = [100][2]float64{}
	for i := 0; i < 100; i++ {
		for j := 0; j < 2; j++ {
			x_data[i][j] = rand.NormFloat64()
		}
	}

	x_dataTF, err := tf.NewTensor(x_data)
	if err != nil {
		panic(err)
	}
	fmt.Println(x_dataTF.Value().([][]float64))
	var w_data = [2][1]float64{{0.1}, {0.2}}
	w_dataTF, err := tf.NewTensor(w_data)
	if err != nil {
		panic(err)
	}
	fmt.Println(w_dataTF)
	////y = x0*0.1 + x1*0.2 + 0.3
	//var y_data = [100]float64{}
	//for i := 0; i < 100; i++ {
	//	y_data[i] = x_data[i][0]*0.1 + x_data[i][1]*0.2 + 0.3
	//}
	//y_dataTF, err := tf.NewTensor(y_data)
	//if err != nil {
	//	panic(err)
	//}
	//fmt.Println(y_dataTF)

	var (
		// Create a graph: y= 0.1*a+0.2+b+0.3
		//
		// Skipping error handling for brevity of this example.
		// The 'op' package can be used to make graph construction code
		// with error handling more succinct.
		g     = tf.NewGraph()
		scope = op.NewScopeWithGraph(g)
		x     = op.Placeholder(scope.SubScope("x"), tf.Double, op.PlaceholderShape(tf.MakeShape(100, 2)))
		W     = op.Placeholder(scope.SubScope("W"), tf.Double, op.PlaceholderShape(tf.MakeShape(2, 1)))
		bias  = op.Placeholder(scope.SubScope("bias"), tf.Double, op.PlaceholderShape(tf.MakeShape(1)))
		mul   = op.MatMul(scope.SubScope("mul"), x, W)
		y     = op.BiasAdd(scope.SubScope("y"), mul, bias)
	)
	writer, _ := os.OpenFile("graph", os.O_WRONLY|os.O_CREATE, 0644)
	//WriteTo 和Import
	g.WriteTo(writer)

	fmt.Println(scope.Err())
	sess, err := tf.NewSession(g, nil)
	if err != nil {
		panic(err)
	}
	defer sess.Close()

	// All the feeds, fetches and targets for subsequent PartialRun.Run
	// calls must be provided at setup.
	pr, err := sess.NewPartialRun(
		[]tf.Output{x, W, bias},
		[]tf.Output{mul, y},
		[]*tf.Operation{y.Op},
	)
	if err != nil {
		panic(err)
	}

	// Feed 'a=1', fetch 'plus2', and compute (but do not fetch) 'plus3'.
	// Imagine this to be the forward pass of unsupervised neural network
	// training of a robot.
	bias_dataTF, _ := tf.NewTensor([1]float64{0.1})
	fetches, err := pr.Run(
		map[tf.Output]*tf.Tensor{x: x_dataTF, W: w_dataTF, bias: bias_dataTF},
		[]tf.Output{mul, y},
		nil)
	if err != nil {
		panic(err)
	}
	v1 := fetches[1].Value().([][]float64)
	v1len := len(v1)
	//v2 := fetches[1].Value().([]float64)

	// Now, feed 'b=4', fetch 'plusB=a+2+3+b'
	// Imagine this to be the result of actuating the robot to determine
	// the error produced by the current state of the neural network.
	fmt.Println(v1len)
	fmt.Println(v1)
}
