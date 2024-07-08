import Area
import Energy
import Latency
import Accuracy

Area.Calculate()    #   Calculate Total Area of the Compute-in-Memory Chip
Latency.Calculate()     #   Calculate Total time for 'Inference' or 'Training'
Accuracy.Calculate()    #   Calculate the Accuracy reflecting Hardware constraints
Energy.Calculate()

Area.Print()
Latency.Print()
Energy.Print()
