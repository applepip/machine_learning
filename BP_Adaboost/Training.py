from Data import *
from BPnetwork import *

print(len(input_train[0]))

model = BPnetwork(input_train, output_train)
result = model.evaluate(input_test, output_test)
print("Test accuracy:", result[1])
