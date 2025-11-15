package predictors

import (
	"ai/utils"
	"fmt"
)

func AndPredictor(inputs []float64, threshold float64) int {
	weights := make([]float64, 0, len(inputs))

	for i := range inputs {
		if inputs[i] > 1.0 {
			inputs[i] = 1.0
		}

		weights = append(weights, 1.0)
	}

	return utils.Perceptron(inputs, weights, threshold)
}

func OrPredictor(inputs []float64, threshold float64) int {
	weights := make([]float64, 0, len(inputs))

	for i := range inputs {
		if inputs[i] > 1.0 {
			inputs[i] = 1.0
		}

		weights = append(weights, 1.0)
	}

	return utils.Perceptron(inputs, weights, threshold)
}

func NandPredictor(inputs []float64, threshold float64) int {
	weights := make([]float64, 0, len(inputs))

	for i := range inputs {
		if inputs[i] > 1.0 {
			inputs[i] = 1.0
		}

		weights = append(weights, -1.0)
	}

	return utils.Perceptron(inputs, weights, threshold)
}

func XorPredictor(inputs []float64) int {
	orPerceptron := OrPredictor(inputs, 0.5)
	nandPerceptron := NandPredictor(inputs, -1.5)

	hiddenLayer := []float64{float64(nandPerceptron), float64(orPerceptron)}

	return AndPredictor(hiddenLayer, 1.5)
}

func TrainAndPredictor() ([]float64, float64) {
	examples := [][]float64{
		{0, 0, 0},
		{0, 0, 1},
		{0, 1, 0},
		{0, 1, 1},
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 0},
		{1, 1, 1},
	}

	labels := []int{0, 0, 0, 0, 0, 0, 0, 1}
	threshold := 0.5

	weights := utils.TrainSinglePerceptron(examples, labels, threshold, 0.1, 100)
	fmt.Printf("\n===AND predictor training results===\n\n")
	utils.PrintTrainingResults(weights, examples, labels, threshold)
	return weights, threshold
}

func TrainOrPredictor() ([]float64, float64) {
	examples := [][]float64{
		{0, 0, 0},
		{0, 0, 1},
		{0, 1, 0},
		{0, 1, 1},
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 0},
		{1, 1, 1},
	}

	labels := []int{0, 1, 1, 1, 1, 1, 1, 1}
	threshold := 0.5

	weights := utils.TrainSinglePerceptron(examples, labels, threshold, 0.1, 100)
	fmt.Printf("\n===OR predictor training results===\n\n")
	utils.PrintTrainingResults(weights, examples, labels, threshold)
	return weights, threshold
}

func TrainNandPredictor() ([]float64, float64) {
	examples := [][]float64{
		{0, 0, 0},
		{0, 0, 1},
		{0, 1, 0},
		{0, 1, 1},
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 0},
		{1, 1, 1},
	}

	labels := []int{1, 1, 1, 1, 1, 1, 1, 0}
	threshold := -0.5

	weights := utils.TrainSinglePerceptron(examples, labels, threshold, 0.1, 100)
	fmt.Printf("\n===NAND predictor training results===\n\n")
	utils.PrintTrainingResults(weights, examples, labels, threshold)
	return weights, threshold
}
