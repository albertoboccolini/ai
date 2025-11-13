package main

import "fmt"

func sign(x float64) int {
	if x >= 0 {
		return 1
	}

	return 0
}

func perceptron(inputs []float64, weights []float64, t float64) int {
	sum := 0.0
	for i := range inputs {
		sum += inputs[i] * weights[i]
	}

	return sign(sum - t)
}

func and_predictor(inputs []float64, t float64) bool {
	weights := make([]float64, 0, len(inputs))
	for i := range inputs {
		if inputs[i] > 1.0 {
			inputs[i] = 1.0
		}

		weights = append(weights, 1.0)
	}

	return perceptron(inputs, weights, t) == 1
}

func or_predictor(inputs []float64, t float64) bool {
	weights := make([]float64, 0, len(inputs))
	for i := range inputs {
		if inputs[i] > 1.0 {
			inputs[i] = 1.0
		}

		weights = append(weights, 1.0)
	}

	return perceptron(inputs, weights, t) == 1
}

func main() {
	input_2_bits := []float64{1.0, 0.0}
	output_2_bits := and_predictor(input_2_bits, 1.5)

	fmt.Printf("is AND (2 bits): %v\n", output_2_bits)

	output_2_bits = or_predictor(input_2_bits, 0.5)

	fmt.Printf("is OR (2 bits): %v\n", output_2_bits)

	input_3_bits := []float64{1.0, 1.0, 1.0}
	output_3_bits := and_predictor(input_3_bits, 2.5)

	fmt.Printf("is AND (3 bits): %v\n", output_3_bits)
}
