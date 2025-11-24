package trainers

import (
	"ai/neural_network"
	"ai/perceptron"
	"fmt"
)

var examples = [][]float64{
	{0, 0, 0},
	{0, 0, 1},
	{0, 1, 0},
	{0, 1, 1},
	{1, 0, 0},
	{1, 0, 1},
	{1, 1, 0},
	{1, 1, 1},
}

func AndTrainer() ([]float64, float64) {
	labels := []int{0, 0, 0, 0, 0, 0, 0, 1}
	threshold := 0.5

	weights := perceptron.Train(examples, labels, threshold, 0.1, 100)
	fmt.Printf("\n===AND predictor training results===\n\n")
	perceptron.PrintTrainingResults(weights, examples, labels, threshold)
	return weights, threshold
}

func OrTrainer() ([]float64, float64) {
	labels := []int{0, 1, 1, 1, 1, 1, 1, 1}
	threshold := 0.5

	weights := perceptron.Train(examples, labels, threshold, 0.1, 100)
	fmt.Printf("\n===OR predictor training results===\n\n")
	perceptron.PrintTrainingResults(weights, examples, labels, threshold)
	return weights, threshold
}

func NandTrainer() ([]float64, float64) {
	labels := []int{1, 1, 1, 1, 1, 1, 1, 0}
	threshold := -0.5

	weights := perceptron.Train(examples, labels, threshold, 0.1, 100)
	fmt.Printf("\n===NAND predictor training results===\n\n")
	perceptron.PrintTrainingResults(weights, examples, labels, threshold)
	return weights, threshold
}

func XorTrainer() *neural_network.NeuralNetwork {
	labels := [][]float64{
		{0}, {1}, {1}, {0}, {1}, {0}, {0}, {1},
	}

	network := neural_network.New(3, 6, 1)
	neural_network.Train(network, examples, labels, 0.5, 10000)

	fmt.Printf("\n===XOR predictor training results===\n\n")
	neural_network.PrintTrainingResults(network, examples, labels)

	return network
}
