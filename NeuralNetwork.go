package Go_NeuralNetwork

// Introductory Neural Network in Go.

func main() {
	goPerceptron := Perceptron{
		input:  [][]float64{{0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 0}},
		Output: []float64{0, 1, 1, 0},
		epochs: 10000,
	}
	goPerceptron.initialize()
	goPerceptron.train()
	print(goPerceptron.forwardProb([]float64{0, 1, 0}), "\n")
	print(goPerceptron.forwardProb([]float64{1, 0, 1}), "\n")
}
