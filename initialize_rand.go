package Go_NeuralNetwork

import (
	"math/rand"
	"time"
)

func (a *Perceptron) initialize() {
	rand.Seed(time.Now().UnixNano())
	a.bias = 0.0
	a.weights = make([]float64, len(a.input[0]))
	for i := 0; i < len(a.input[0]); i++ {
		a.weights[i] = rand.Float64()
	}
}
