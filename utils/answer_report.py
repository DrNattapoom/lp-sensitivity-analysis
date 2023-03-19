import numpy as np
import pandas as pd

class AnswerReport:
    """
    A class provides methods for generating amswer reports for decision variables and constraints in a linear programming model.
    """

    # set pandas display format
    pd.set_option('display.float_format', str)

    def __init__(self, model) -> None:
        """
        Initializes a new instance of the AnswerReport class.
        
        Args:
            model (docplex.mp.model.Model): A linear programming model to analyze.
        """
        self.model = model
    
    def decision_variables(self) -> pd.DataFrame:
        """
        Get an answer report for the decision variables in a linear programming model.

        Returns:
            pd.DataFrame: A table with information on each variable, including final value, and variable type
        """
        cpx = self.model.get_engine().get_cplex()
        names = cpx.variables.get_names()
        final_values = self.model.solution.get_values(self.model.iter_variables())
        variable_types = [variable._vartype.short_name for variable in self.model.iter_variables()]
        return pd.DataFrame({
            'Name' : names,
            'Final Value' : final_values,
            'Variable Type' : variable_types
        }).set_index('Name')
    
    def constraints(self) -> pd.DataFrame:
        """
        Get an answer report for the constraints in a linear programming model.

        Returns:
            pd.DataFrame: A table with information on each constraint, including status, and slack
        """
        cpx = self.model.get_engine().get_cplex()
        names = cpx.linear_constraints.get_names()
        slacks = cpx.solution.get_linear_slacks()
        statuses = [f"{'non-'*(slack != 0)}binding" for slack in slacks]
        return pd.DataFrame({
            'Name' : names,
            'Status' : statuses,
            'Slack' : slacks
        }).set_index('Name')
