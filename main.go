package main

import (
	"fmt"
	"go-ai/predictors"
	"go-ai/utils"
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

	andWeights, andThreshold := predictors.TrainAndPredictor()
	orWeights, orThreshold := predictors.TrainOrPredictor()
	nandWeights, nandThreshold := predictors.TrainNandPredictor()

	inputs3Bits := []float64{1.0, 0.0, 1.0}

	outputAnd = utils.Perceptron(inputs3Bits, andWeights, andThreshold)
	outputOr = utils.Perceptron(inputs3Bits, orWeights, orThreshold)
	outputNand = utils.Perceptron(inputs3Bits, nandWeights, nandThreshold)

	fmt.Printf("\n%v is AND: %v\n", inputs3Bits, outputAnd == 1)
	fmt.Printf("%v is OR: %v\n", inputs3Bits, outputOr == 1)
	fmt.Printf("%v is NAND: %v\n", inputs3Bits, outputNand == 1)
}
