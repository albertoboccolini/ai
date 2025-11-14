package main

import "fmt"

func sign(x float64) int {
	if x >= 0 {
		return 1
	}

	return 0
}

func perceptron(inputs []float64, weights []float64, threshold float64) int {
	sum := 0.0
	for i := range inputs {
		sum += inputs[i] * weights[i]
	}

	return sign(sum - threshold)
}

func and_predictor(inputs []float64, threshold float64) int {
	weights := make([]float64, 0, len(inputs))
	for i := range inputs {
		if inputs[i] > 1.0 {
			inputs[i] = 1.0
		}

		weights = append(weights, 1.0)
	}

	return perceptron(inputs, weights, threshold)
}

func or_predictor(inputs []float64, threshold float64) int {
	weights := make([]float64, 0, len(inputs))
	for i := range inputs {
		if inputs[i] > 1.0 {
			inputs[i] = 1.0
		}

		weights = append(weights, 1.0)
	}

	return perceptron(inputs, weights, threshold)
}

func nand_predictor(inputs []float64, threshold float64) int {
	weights := make([]float64, 0, len(inputs))
	for i := range inputs {
		if inputs[i] > 1.0 {
			inputs[i] = 1.0
		}

		weights = append(weights, -1.0)
	}

	return perceptron(inputs, weights, threshold)
}

func xor_predictor(inputs []float64) int {
	or_perceptron := or_predictor(inputs, 0.5)
	nand_perceptron := nand_predictor(inputs, -1.5)

	hidden_layer := []float64{float64(nand_perceptron), float64(or_perceptron)}

	return and_predictor(hidden_layer, 1.5)
}

func main() {
	inputs := []float64{1.0, 0.0}
	output := xor_predictor(inputs)

	fmt.Printf("is XOR: %v\n", output == 1)
}
