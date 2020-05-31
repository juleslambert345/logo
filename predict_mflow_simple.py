
from os.path import join
import mlflow.pytorch
number_logo =2




generator = mlflow.pyfunc.load_model(join('experiements_result', '2020_05_31__10_15_test3'))

img = generator.predict(number_logo)

#img = return_logos(2, join('simple_save'))
print(img)