from grader_contracts.pandas_tasks import TitanicInput
from grader_contracts.correlation_tasks import BrainDataInput
from pandas_tasks import analyze_titanic
from correlation_tasks import analyze_brain_correlations

print(analyze_titanic(TitanicInput(csv_path="titanic.csv")))
print(analyze_brain_correlations(BrainDataInput(csv_path="brainsize.txt")))