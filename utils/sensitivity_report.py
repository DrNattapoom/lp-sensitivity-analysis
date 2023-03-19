import numpy as np
import pandas as pd

class SensitivityReport:
    """
    A class provides methods for generating sensitivity reports for decision variables and constraints in a linear programming model.
    """

    # set pandas display format
    pd.set_option('display.float_format', str)

    def __init__(self, model) -> None:
        """
        Initializes a new instance of the SensitivityReport class.
        
        Args:
            model (docplex.mp.model.Model): A linear programming model to analyze.
        """
        self.model = model

    def get_allowable_changes(self, current_values: np.ndarray, bounds: list) -> tuple:
        """
        Get the allowable changes in the values of variables.

        Args:
            current_values (np.ndarray): A 1-dimensional numpy array containing the current values of the variables.
            bounds (list): A list of tuples containing the lower and upper bounds for each variable.

        Returns:
            tuple: A tuple containing two 1-dimensional numpy arrays. The first array contains the allowable decreases for each variable, and the second array contains the allowable increases for each variable.
        """
        lower_bounds, upper_bounds = zip(*bounds)
        allowable_decreases = current_values - lower_bounds
        allowable_increases = upper_bounds - current_values
        return allowable_decreases, allowable_increases
    
    def decision_variables(self) -> pd.DataFrame:
        """
        Get a sensitivity report for the decision variables in a linear programming model.

        Returns:
            pd.DataFrame: A table with information on each variable, including final value, reduced cost, objective coefficient, allowable increase, and allowable decrease.
        """
        cpx = self.model.get_engine().get_cplex()
        names = cpx.variables.get_names()
        final_values = cpx.solution.get_values()
        reduced_costs = cpx.solution.get_reduced_costs()
        objective_coefficients = np.array(list(map(
            lambda name : self.model.objective_coef(
                self.model.get_var_by_name(name)
            ),
            names
        )))
        allowable_decreases, allowable_increases = self.get_allowable_changes(
            objective_coefficients,
            cpx.solution.sensitivity.objective()
        )
        return pd.DataFrame({
            'Name' : names,
            'Final Value' : final_values,
            'Reduced Cost' : reduced_costs,
            'Objective Coefficient': objective_coefficients,
            'Allowable Increase' : allowable_increases,
            'Allowable Decrease' : allowable_decreases
        }).set_index('Name')
    
    def constraints(self) -> pd.DataFrame:
        """
        Get a sensitivity report for the constraints in a linear programming model.

        Returns:
            pd.DataFrame: A table with information on each constraint, including final value, shadow price, objective coefficient, allowable increase, and allowable decrease.
        """
        cpx = self.model.get_engine().get_cplex()
        names = cpx.linear_constraints.get_names()
        rhs = np.array([
            self.model.get_constraint_by_name(name).rhs.get_constant()
            for name in names
        ])
        final_values = rhs - cpx.solution.get_linear_slacks()
        shadow_prices = cpx.solution.get_dual_values()
        allowable_decreases, allowable_increases = self.get_allowable_changes(
            rhs,
            cpx.solution.sensitivity.rhs()
        )
        return pd.DataFrame({
            'Name' : names, 
            'Final Value' : final_values,
            'Shadow Price' : shadow_prices,
            'Constraint RHS': rhs, 
            'Allowable Increase' : allowable_increases, 
            'Allowable Decrease' : allowable_decreases  
        }).set_index('Name')
