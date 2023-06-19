# GO 和 tensorflow

## 写在前面 @20230614

    golang语言是我本职编程语言(c/c++)外，比较喜欢的一门编程语言，好的出身(google产品)加上令人欣喜的
    设计理念(在性能和功能上都兼顾)，不得不令人对它的前景充满信心。

    tensorflow 同样也是google领导的机器学习的开源解决方案。开源是个好方案，集大成者。同样机器学习也是
    一门这样的学问：没有完美的机器学习模型，它是数学和编程的完美体，同时需要大量的数据和人员来反复的修正。
    但tensorflow给机器学习的初学者们提供一个便捷的入门，可以稍微的跳过那些繁杂晦涩的数学公式。虽然这些公式
    是必须的，但作为使用者，我们往往只要在先哲们的带领下正确的使用即可。(这里不得不说，万物相通，不管是狼群还是
    羊群，都得有一个有力的领头者，带领它们生存，个体的、弱小的只能期许领头者英明睿智，才不使整个族群陷入死地)
    但是比较令人失望的是，tensorflow到后面竟没有专门的为golang来封装接口，就好像同一个家庭的孩子，互相之间
    不通往来一样，想不明白。这里只能期望google大佬们能开恩了。

    话归正题，本工程的意图是通过学习网上的开源代码，来建立自己的go-tensorflow解决方案。加油吧，别被时代优化了。

    初期的几个工程，是从最近的书籍上摘取的部分章节的工程，主要golang tensorflow的，意图是通过这些来学习到用golang
    tensorflow库，来完成机器学习的编程。内容主要涉及：数据初步分析，数据清洗与整理、调用tensorflow现成的模型进行机器学习。
    并以此来完成自己使用机器学习库的目的。当然能想到的是，基于自己的主编程语言c++，还应该开辟一个opencv的学习代码。

## linear linear_regression

    用tesnsorflow库做线性运算，y=W*x+bias
    tensorflow内是数据类型和graph session等外部结构，op库是放置图上的节点，及其操作
    这块的延伸应该是结合tensor的例程来做，但是可惜的是都是python的
    这里可以看出go 版本下的tensorflow库，只能正向输出，不能做训练。
    对照实际的tensorflow模型，调用的时候可以根据节点的名称来设置Run的输入和输出，但是golang自己建立节点的时候，
    有两种做法：
    x, _ := graph.AddOperation(tf.OpSpec{Type: "Placeholder", Name: "x", Attrs: map[string]interface{}{
		"dtype": tf.Float,
	}})
        这样的有节点名称，但是对加入操作十分不友好

    scope := op.NewScopeWithGraph(graph)
	mul := op.Mul(scope, x.Output(0), w.Output(0))

        这样是可以方便的通过scope来加入操作，但是节点名称设置不了。

## deepLearning

    这个工程和tensorflow的go下example_inception_inference_test.go一模一样

## faceFinder

    pigo "github.com/esimov/pigo/core" 主要是利用这个库来识别图片

## audio-example

    
## realtimeObjectDetection

    