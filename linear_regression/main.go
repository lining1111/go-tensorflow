package main

import (
	"fmt"
	tf "github.com/wamuir/graft/tensorflow"
	"github.com/wamuir/graft/tensorflow/op"
	"math/rand"
)

func main() {
	//随机生产一些数据
	var trainData []float32
	var trainLabels []float32
	for i := 0; i < 1000; i++ {
		trainData = append(trainData, float32(rand.Intn(100)))
		trainLabels = append(trainLabels, trainData[i]*0.3+5)
	}

	//创建Graph
	graph := tf.NewGraph()
	//设置模型输入和输出
	//input, _ := tf.NewTensor([1][1]float32{{0.0}})
	x, _ := graph.AddOperation(tf.OpSpec{Type: "Placeholder", Name: "x", Attrs: map[string]interface{}{
		"dtype": tf.Float,
	}})
	y, _ := graph.AddOperation(tf.OpSpec{Type: "Placeholder", Name: "y", Attrs: map[string]interface{}{
		"dtype": tf.Float,
	}})
	fmt.Println(y.Name())
	w, _ := graph.AddOperation(tf.OpSpec{Type: "Placeholder", Name: "w", Attrs: map[string]interface{}{
		"dtype": tf.Float,
	}})
	//w := op.Const(scope.SubScope("w"), float32(0.3))
	scope := op.NewScopeWithGraph(graph)
	mul := op.Mul(scope, x.Output(0), w.Output(0))
	add := op.Add(scope, mul, op.Const(scope, float32(5)))

	//assignAdd := op.AssignAddVariableOp(scope.SubScope("assign_add"), y, add)
	fmt.Println(scope.Err())

	//创建Session执行graph
	session, _ := tf.NewSession(graph, nil)
	defer session.Close()
	//训练模型
	for i := 0; i < 1000; i++ {
		feed_x, _ := tf.NewTensor(trainData[i])
		//feed_y, _ := tf.NewTensor([][]float32{{trainLabels[i]}})
		wvalue, _ := tf.NewTensor(float32(0.5))
		fecths, _ := session.Run(
			map[tf.Output]*tf.Tensor{
				graph.Operation("x").Output(0): feed_x,
				graph.Operation("w").Output(0): wvalue,
			},
			[]tf.Output{
				mul,
				add,
			},
			[]*tf.Operation{add.Op})
		fmt.Println(fecths[0].Value().(float32))
	}
	//预测结果
	feed_10, _ := tf.NewTensor([][]float32{{10}})
	feed_05w, _ := tf.NewTensor([][]float32{{0.5}})
	feed_0y, _ := tf.NewTensor([1][1]float32{{0.0}})
	fecths, _ := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("x").Output(0): feed_10,
			graph.Operation("w").Output(0): feed_05w,
			graph.Operation("y").Output(0): feed_0y,
		},
		[]tf.Output{
			add,
		},
		[]*tf.Operation{add.Op})

	result := fecths[0].Value().([][]float32)[0][0]
	fmt.Println(result)
}
