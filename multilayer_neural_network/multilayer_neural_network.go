package multilayer_neural_network

import (
	"fmt"
	"math"
	"math/rand"
)

type MultilayerNeuralNetwork struct {
	InputFeatures              int
	HiddenLayerNeurons         int
	OutputFeatures             int
	WeightsInputToHiddenLayer  [][]float64
	WeightsHiddenToOutputLayer [][]float64
	BiasForHiddenLayer         []float64
	BiasForOutputLayer         []float64
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

func New(inputFeatures, hiddenLayerNeurons, outputFeatures int) *MultilayerNeuralNetwork {
	network := &MultilayerNeuralNetwork{
		InputFeatures:              inputFeatures,
		HiddenLayerNeurons:         hiddenLayerNeurons,
		OutputFeatures:             outputFeatures,
		WeightsInputToHiddenLayer:  make([][]float64, inputFeatures),
		WeightsHiddenToOutputLayer: make([][]float64, hiddenLayerNeurons),
		BiasForHiddenLayer:         make([]float64, hiddenLayerNeurons),
		BiasForOutputLayer:         make([]float64, outputFeatures),
	}

	for i := range network.WeightsInputToHiddenLayer {
		network.WeightsInputToHiddenLayer[i] = make([]float64, hiddenLayerNeurons)

		for j := range network.WeightsInputToHiddenLayer[i] {
			network.WeightsInputToHiddenLayer[i][j] = rand.Float64()*2 - 1
		}
	}

	for i := range network.WeightsHiddenToOutputLayer {
		network.WeightsHiddenToOutputLayer[i] = make([]float64, outputFeatures)

		for j := range network.WeightsHiddenToOutputLayer[i] {
			network.WeightsHiddenToOutputLayer[i][j] = rand.Float64()*2 - 1
		}
	}

	for i := range network.BiasForHiddenLayer {
		network.BiasForHiddenLayer[i] = rand.Float64()*2 - 1
	}

	for i := range network.BiasForOutputLayer {
		network.BiasForOutputLayer[i] = rand.Float64()*2 - 1
	}

	return network
}

func forward(network *MultilayerNeuralNetwork, inputs []float64) ([]float64, []float64) {
	hiddenLayerOutputs := make([]float64, network.HiddenLayerNeurons)
	for hiddenNeuronIndex := range hiddenLayerOutputs {
		hiddenNeuronInput := network.BiasForHiddenLayer[hiddenNeuronIndex]

		for inputIndex := range inputs {
			hiddenNeuronInput += inputs[inputIndex] * network.WeightsInputToHiddenLayer[inputIndex][hiddenNeuronIndex]
		}
		hiddenLayerOutputs[hiddenNeuronIndex] = sigmoid(hiddenNeuronInput)
	}

	outputLayerOutputs := make([]float64, network.OutputFeatures)
	for outputNeuronIndex := range outputLayerOutputs {
		outputNeuronInput := network.BiasForOutputLayer[outputNeuronIndex]

		for hiddenNeuronIndex := range hiddenLayerOutputs {
			outputNeuronInput += hiddenLayerOutputs[hiddenNeuronIndex] * network.WeightsHiddenToOutputLayer[hiddenNeuronIndex][outputNeuronIndex]
		}
		outputLayerOutputs[outputNeuronIndex] = sigmoid(outputNeuronInput)
	}

	return hiddenLayerOutputs, outputLayerOutputs
}

func Train(network *MultilayerNeuralNetwork, examples [][]float64, labels [][]float64, lambda float64, maxEpochs int) {
	for range maxEpochs {
		for exampleIndex := range examples {
			hiddenLayerOutputs, outputLayerOutputs := forward(network, examples[exampleIndex])

			outputLayerErrors := make([]float64, network.OutputFeatures)
			for outputIndex := range outputLayerOutputs {
				error := labels[exampleIndex][outputIndex] - outputLayerOutputs[outputIndex]
				outputLayerErrors[outputIndex] = error * sigmoidDerivative(outputLayerOutputs[outputIndex])
			}

			hiddenLayerErrors := make([]float64, network.HiddenLayerNeurons)
			for hiddenIndex := range hiddenLayerOutputs {
				errorSum := 0.0
				for outputIndex := range outputLayerOutputs {
					errorSum += network.WeightsHiddenToOutputLayer[hiddenIndex][outputIndex] * outputLayerErrors[outputIndex]
				}
				hiddenLayerErrors[hiddenIndex] = sigmoidDerivative(hiddenLayerOutputs[hiddenIndex]) * errorSum
			}

			for hiddenIndex := range network.WeightsHiddenToOutputLayer {
				for outputIndex := range network.WeightsHiddenToOutputLayer[hiddenIndex] {
					network.WeightsHiddenToOutputLayer[hiddenIndex][outputIndex] += lambda * outputLayerErrors[outputIndex] * hiddenLayerOutputs[hiddenIndex]
				}
			}

			for outputIndex := range network.BiasForOutputLayer {
				network.BiasForOutputLayer[outputIndex] += lambda * outputLayerErrors[outputIndex]
			}

			for inputIndex := range network.WeightsInputToHiddenLayer {
				for hiddenIndex := range network.WeightsInputToHiddenLayer[inputIndex] {
					network.WeightsInputToHiddenLayer[inputIndex][hiddenIndex] += lambda * hiddenLayerErrors[hiddenIndex] * examples[exampleIndex][inputIndex]
				}
			}

			for hiddenIndex := range network.BiasForHiddenLayer {
				network.BiasForHiddenLayer[hiddenIndex] += lambda * hiddenLayerErrors[hiddenIndex]
			}
		}
	}
}

func Predict(network *MultilayerNeuralNetwork, inputs []float64) []float64 {
	_, outputs := forward(network, inputs)
	return outputs
}

func PrintTrainingResults(network *MultilayerNeuralNetwork, examples [][]float64, labels [][]float64) {
	fmt.Println("Weights Input -> Hidden:")
	for i := range network.WeightsInputToHiddenLayer {
		fmt.Printf("  Input %d: %v\n", i, network.WeightsInputToHiddenLayer[i])
	}
	fmt.Printf("Bias Hidden: %v\n\n", network.BiasForHiddenLayer)

	fmt.Println("Weights Hidden -> Output:")
	for i := range network.WeightsHiddenToOutputLayer {
		fmt.Printf("  Hidden %d: %v\n", i, network.WeightsHiddenToOutputLayer[i])
	}
	fmt.Printf("Bias Output: %v\n\n", network.BiasForOutputLayer)

	for i, example := range examples {
		output := Predict(network, example)
		result := 0
		if output[0] > 0.5 {
			result = 1
		}
		fmt.Printf("Input: %v, Output: %.4f, Predicted: %d, Expected: %.0f\n",
			example, output[0], result, labels[i][0])
	}
}
