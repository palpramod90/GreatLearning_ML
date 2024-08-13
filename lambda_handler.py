import athena_udf
import pandas as pd
from pyarrow import Schema
import jellyfish


class JaroUDF(athena_udf.BaseAthenaUDF):

    @staticmethod
    def handle_athena_record(input_schema, output_schema, arguments: list[Any]):
        if len(arguments) < 2:
            raise ValueError("insuffient parameters")
        string1 = arguments[0]
        string2 =arguments[1]
        similarity = jellyfish.jaro_winkler_similarity(string1,string2)
        return similarity


lambda_handler = JaroUDF().lambda_handler
