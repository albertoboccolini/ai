package neural_network

import (
	"fmt"
	"math"
	"math/rand"
)

type NeuralNetwork struct {
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

func New(inputFeatures, hiddenLayerNeurons, outputFeatures int) *NeuralNetwork {
	network := &NeuralNetwork{
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

func forward(network *NeuralNetwork, inputs []float64) ([]float64, []float64) {
	hiddenLayerOutputs := make([]float64, network.HiddenLayerNeurons)
	for hiddenLayerOutput := range hiddenLayerOutputs {
		hiddenNeuronInput := network.BiasForHiddenLayer[hiddenLayerOutput]

		for input := range inputs {
			hiddenNeuronInput += inputs[input] * network.WeightsInputToHiddenLayer[input][hiddenLayerOutput]
		}
		hiddenLayerOutputs[hiddenLayerOutput] = sigmoid(hiddenNeuronInput)
	}

	outputLayerOutputs := make([]float64, network.OutputFeatures)
	for outputNeuron := range outputLayerOutputs {
		outputNeuronInput := network.BiasForOutputLayer[outputNeuron]

		for hiddenLayerOutput := range hiddenLayerOutputs {
			outputNeuronInput += hiddenLayerOutputs[hiddenLayerOutput] * network.WeightsHiddenToOutputLayer[hiddenLayerOutput][outputNeuron]
		}
		outputLayerOutputs[outputNeuron] = sigmoid(outputNeuronInput)
	}

	return hiddenLayerOutputs, outputLayerOutputs
}

func Train(network *NeuralNetwork, examples [][]float64, labels [][]float64, lambda float64, maxEpochs int) {
	for range maxEpochs {
		for example := range examples {
			hiddenLayerOutputs, outputLayerOutputs := forward(network, examples[example])

			outputLayerErrors := make([]float64, network.OutputFeatures)
			for output := range outputLayerOutputs {
				error := labels[example][output] - outputLayerOutputs[output]
				outputLayerErrors[output] = error * sigmoidDerivative(outputLayerOutputs[output])
			}

			hiddenLayerErrors := make([]float64, network.HiddenLayerNeurons)
			for hiddenLayerOutput := range hiddenLayerOutputs {
				errorSum := 0.0
				for outputLayerOutput := range outputLayerOutputs {
					errorSum += network.WeightsHiddenToOutputLayer[hiddenLayerOutput][outputLayerOutput] * outputLayerErrors[outputLayerOutput]
				}

				hiddenLayerErrors[hiddenLayerOutput] = sigmoidDerivative(hiddenLayerOutputs[hiddenLayerOutput]) * errorSum
			}

			for hiddenIndex := range network.WeightsHiddenToOutputLayer {
				for outputIndex := range network.WeightsHiddenToOutputLayer[hiddenIndex] {
					network.WeightsHiddenToOutputLayer[hiddenIndex][outputIndex] += lambda * outputLayerErrors[outputIndex] * hiddenLayerOutputs[hiddenIndex]
				}
			}

			for output := range network.BiasForOutputLayer {
				network.BiasForOutputLayer[output] += lambda * outputLayerErrors[output]
			}

			for input := range network.WeightsInputToHiddenLayer {
				for hiddenLayer := range network.WeightsInputToHiddenLayer[input] {
					network.WeightsInputToHiddenLayer[input][hiddenLayer] += lambda * hiddenLayerErrors[hiddenLayer] * examples[example][input]
				}
			}

			for hiddenLayer := range network.BiasForHiddenLayer {
				network.BiasForHiddenLayer[hiddenLayer] += lambda * hiddenLayerErrors[hiddenLayer]
			}
		}
	}
}

func Predict(network *NeuralNetwork, inputs []float64) []float64 {
	_, outputs := forward(network, inputs)
	return outputs
}

func PrintTrainingResults(network *NeuralNetwork, examples [][]float64, labels [][]float64) {
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
