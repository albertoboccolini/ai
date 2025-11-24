package main

import (
	"ai/neural_network"
	"ai/perceptron"
	"ai/predictors"
	"ai/trainers"
	"fmt"
)

func main() {
	inputs2Bits := []float64{1.0, 1.0}

	outputAnd := predictors.AndPredictor(inputs2Bits, 1.5)
	outputOr := predictors.OrPredictor(inputs2Bits, 0.5)
	outputNand := predictors.NandPredictor(inputs2Bits, -1.5)
	outputXor := predictors.XorPredictor(inputs2Bits)

	fmt.Printf("%v is AND: %v\n", inputs2Bits, outputAnd == 1)
	fmt.Printf("%v is OR: %v\n", inputs2Bits, outputOr == 1)
	fmt.Printf("%v is NAND: %v\n", inputs2Bits, outputNand == 1)
	fmt.Printf("%v is XOR: %v\n", inputs2Bits, outputXor == 1)

	andWeights, andThreshold := trainers.AndTrainer()
	orWeights, orThreshold := trainers.OrTrainer()
	nandWeights, nandThreshold := trainers.NandTrainer()

	inputs3Bits := []float64{1.0, 0.0, 1.0}

	outputAnd = perceptron.Perceptron(inputs3Bits, andWeights, andThreshold)
	outputOr = perceptron.Perceptron(inputs3Bits, orWeights, orThreshold)
	outputNand = perceptron.Perceptron(inputs3Bits, nandWeights, nandThreshold)

	fmt.Printf("\n%v is AND: %v\n", inputs3Bits, outputAnd == 1)
	fmt.Printf("%v is OR: %v\n", inputs3Bits, outputOr == 1)
	fmt.Printf("%v is NAND: %v\n", inputs3Bits, outputNand == 1)

	xorNetwork := trainers.XorTrainer()
	outputXorNetwork := neural_network.Predict(xorNetwork, inputs3Bits)
	fmt.Printf("\n%v is XOR: %v\n", inputs3Bits, outputXorNetwork[0] > 0.5)
}
