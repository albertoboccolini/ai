package predictors

import (
	"ai/single_perceptron"
)

func AndPredictor(inputs []float64, threshold float64) int {
	weights := make([]float64, 0, len(inputs))

	for i := range inputs {
		if inputs[i] > 1.0 {
			inputs[i] = 1.0
		}

		weights = append(weights, 1.0)
	}

	return single_perceptron.Perceptron(inputs, weights, threshold)
}

func OrPredictor(inputs []float64, threshold float64) int {
	weights := make([]float64, 0, len(inputs))

	for i := range inputs {
		if inputs[i] > 1.0 {
			inputs[i] = 1.0
		}

		weights = append(weights, 1.0)
	}

	return single_perceptron.Perceptron(inputs, weights, threshold)
}

func NandPredictor(inputs []float64, threshold float64) int {
	weights := make([]float64, 0, len(inputs))

	for i := range inputs {
		if inputs[i] > 1.0 {
			inputs[i] = 1.0
		}

		weights = append(weights, -1.0)
	}

	return single_perceptron.Perceptron(inputs, weights, threshold)
}

func XorPredictor(inputs []float64) int {
	orPerceptron := OrPredictor(inputs, 0.5)
	nandPerceptron := NandPredictor(inputs, -1.5)

	hiddenLayer := []float64{float64(nandPerceptron), float64(orPerceptron)}

	return AndPredictor(hiddenLayer, 1.5)
}
