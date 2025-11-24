package perceptron

import (
	"fmt"
)

func sign(x float64) int {
	if x >= 0 {
		return 1
	}

	return 0
}

func Perceptron(inputs []float64, weights []float64, threshold float64) int {
	sum := 0.0

	for i := range inputs {
		sum += inputs[i] * weights[i]
	}

	return sign(sum - threshold)
}

func Train(examples [][]float64, labels []int, threshold float64, lambda float64, maxEpochs int) []float64 {
	weights := make([]float64, len(examples[0]))

	for range maxEpochs {
		epoch_errors := 0
		for i, xi := range examples {
			yi := labels[i]

			prediction := Perceptron(xi, weights, threshold)
			prediction_error := yi - prediction

			if prediction_error != 0 {
				epoch_errors++

				for j := range weights {
					weights[j] += lambda * float64(prediction_error) * xi[j]
				}
			}
		}

		if epoch_errors == 0 {
			break
		}
	}

	return weights
}

func PrintTrainingResults(weights []float64, examples [][]float64, labels []int, threshold float64) {
	fmt.Println("Weights:", weights)

	for i, example := range examples {
		output := Perceptron(example, weights, threshold)
		fmt.Printf("Input: %v, Output: %d, Expected: %d\n", example, output, labels[i])
	}
}
