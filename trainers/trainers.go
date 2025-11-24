package trainers

import (
	"ai/multilayer_neural_network"
	"ai/single_perceptron"
	"fmt"
)

var examples = [][]float64{
	{0, 0, 0},
	{0, 0, 1},
	{0, 1, 0},
	{0, 1, 1},
	{1, 0, 0},
	{1, 1, 1},
}

func AndTrainer() ([]float64, float64) {

	labels := []int{0, 0, 0, 0, 0, 1}
	threshold := 0.5

	weights := single_perceptron.Train(examples, labels, threshold, 0.1, 100)
	fmt.Printf("\n===AND predictor training results===\n\n")
	single_perceptron.PrintTrainingResults(weights, examples, labels, threshold)
	return weights, threshold
}

func OrTrainer() ([]float64, float64) {
	labels := []int{0, 1, 1, 1, 1, 1}
	threshold := 0.5

	weights := single_perceptron.Train(examples, labels, threshold, 0.1, 100)
	fmt.Printf("\n===OR predictor training results===\n\n")
	single_perceptron.PrintTrainingResults(weights, examples, labels, threshold)
	return weights, threshold
}

func NandTrainer() ([]float64, float64) {
	labels := []int{1, 1, 1, 1, 1, 0}
	threshold := -0.5

	weights := single_perceptron.Train(examples, labels, threshold, 0.1, 100)
	fmt.Printf("\n===NAND predictor training results===\n\n")
	single_perceptron.PrintTrainingResults(weights, examples, labels, threshold)
	return weights, threshold
}

func XorTrainer() *multilayer_neural_network.MultilayerNeuralNetwork {
	labels := [][]float64{
		{0},
		{1},
		{1},
		{0},
		{1},
		{0},
	}

	network := multilayer_neural_network.New(3, 6, 1)
	multilayer_neural_network.Train(network, examples, labels, 0.5, 5000)

	fmt.Printf("\n===XOR predictor training results===\n\n")
	multilayer_neural_network.PrintTrainingResults(network, examples, labels)

	return network
}
