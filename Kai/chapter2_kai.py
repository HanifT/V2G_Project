import pandas as pd
from kai_read import (read_clean_kai, extract_datetime_components, rename_col)

data = read_clean_kai()
data = extract_datetime_components(data)
data = rename_col(data)


