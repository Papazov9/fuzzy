"""
Fuzzy Expert System for Evaluating Student Academic Performance
================================================================

This module implements a fuzzy expert system that evaluates student academic 
performance based on multiple input criteria using the compositional rule of 
inference and centroid defuzzification method.

Course: Fuzzy Sets and Applications
Topic: Design and implementation of a fuzzy expert system for evaluating 
       student academic performance based on precise and expert knowledge,
       using the compositional rule of inference.

Author: Student
Date: February 2026
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 1: DEFINITION OF FUZZY VARIABLES AND MEMBERSHIP FUNCTIONS
# =============================================================================

class FuzzyStudentEvaluator:
    """
    Fuzzy Expert System for Student Academic Performance Evaluation.
    
    This class implements a complete fuzzy inference system with:
    - Four input variables: GPA, Absences, Participation, Test Score
    - One output variable: Overall Performance
    - Membership functions for all linguistic terms
    - Rule base with 12 expert-defined rules
    - Mamdani inference with centroid defuzzification
    
    Attributes:
        gpa: Fuzzy variable for Grade Point Average (0-6)
        absences: Fuzzy variable for number of absences (0-30)
        participation: Fuzzy variable for class participation (0-100)
        test_score: Fuzzy variable for test score (0-100)
        performance: Fuzzy output variable for overall performance (0-100)
        rules: List of fuzzy rules defining expert knowledge
        system: The complete fuzzy control system
        simulator: Simulation object for computing outputs
    """
    
    def __init__(self):
        """Initialize the fuzzy expert system with all variables and rules."""
        self._define_input_variables()
        self._define_output_variable()
        self._define_rules()
        self._build_control_system()
    
    def __getitem__(self, key: str):
        """
        Support dict-like access for backward compatibility.
        Allows fuzzy_sys['gpa'], fuzzy_sys['system'], etc.
        """
        if key == 'system':
            return self.system
        elif hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Unknown key: {key}")
        
    def _define_input_variables(self):
        """
        Define all fuzzy input variables with their universes of discourse
        and membership functions.
        
        Input Variables:
        ----------------
        1. GPA (Grade Point Average): Range [0, 6]
           - Linguistic terms: low, medium, high
           - Domain interpretation: University grading scale
           
        2. Absences: Range [0, 30]
           - Linguistic terms: few, moderate, many
           - Domain interpretation: Number of missed classes per semester
           
        3. Participation: Range [0, 100]
           - Linguistic terms: passive, moderate, active
           - Domain interpretation: Percentage score for class engagement
           
        4. Test Score: Range [0, 100]
           - Linguistic terms: poor, average, good, excellent
           - Domain interpretation: Percentage score on examinations
        """
        
        # ---------------------------------------------------------------------
        # INPUT VARIABLE 1: Grade Point Average (GPA)
        # Universe of discourse: [2, 6] - Bulgarian grading scale
        # Linguistic terms: low, medium, high
        # Note: In Bulgaria, grades range from 2 (fail) to 6 (excellent)
        # ---------------------------------------------------------------------
        self.gpa = ctrl.Antecedent(np.arange(2, 6.1, 0.1), 'gpa')
        
        # Membership functions using triangular (trimf) shapes
        # Low GPA: 2-3.5 (failing to poor)
        self.gpa['low'] = fuzz.trimf(self.gpa.universe, [2, 2, 3.5])
        # Medium GPA: 3-5 (satisfactory to good)
        self.gpa['medium'] = fuzz.trimf(self.gpa.universe, [3, 4, 5])
        # High GPA: 4.5-6 (good to excellent)
        self.gpa['high'] = fuzz.trimf(self.gpa.universe, [4.5, 6, 6])
        
        # ---------------------------------------------------------------------
        # INPUT VARIABLE 2: Number of Absences
        # Universe of discourse: [0, 30]
        # Linguistic terms: few, moderate, many
        # Note: Lower absences are better for performance
        # ---------------------------------------------------------------------
        self.absences = ctrl.Antecedent(np.arange(0, 31, 1), 'absences')
        
        # Few absences: 0-10 (good attendance)
        self.absences['few'] = fuzz.trimf(self.absences.universe, [0, 0, 10])
        # Moderate absences: 5-25 (acceptable attendance)
        self.absences['moderate'] = fuzz.trimf(self.absences.universe, [5, 15, 25])
        # Many absences: 20-30 (poor attendance)
        self.absences['many'] = fuzz.trimf(self.absences.universe, [20, 30, 30])
        
        # ---------------------------------------------------------------------
        # INPUT VARIABLE 3: Class Participation Level
        # Universe of discourse: [0, 100]
        # Linguistic terms: passive, moderate, active
        # ---------------------------------------------------------------------
        self.participation = ctrl.Antecedent(np.arange(0, 101, 1), 'participation')
        
        # Passive participation: 0-40 (minimal engagement)
        self.participation['passive'] = fuzz.trimf(self.participation.universe, [0, 0, 40])
        # Moderate participation: 30-70 (average engagement)
        self.participation['moderate'] = fuzz.trimf(self.participation.universe, [30, 50, 70])
        # Active participation: 60-100 (high engagement)
        self.participation['active'] = fuzz.trimf(self.participation.universe, [60, 100, 100])
        
        # ---------------------------------------------------------------------
        # INPUT VARIABLE 4: Test Score
        # Universe of discourse: [0, 100]
        # Linguistic terms: poor, average, good, excellent
        # Using trapezoidal membership functions for more nuanced evaluation
        # ---------------------------------------------------------------------
        self.test_score = ctrl.Antecedent(np.arange(0, 101, 1), 'test_score')
        
        # Poor score: 0-50 (failing or near-failing)
        self.test_score['poor'] = fuzz.trapmf(self.test_score.universe, [0, 0, 30, 50])
        # Average score: 40-80 (passing but not exceptional)
        self.test_score['average'] = fuzz.trimf(self.test_score.universe, [40, 60, 80])
        # Good score: 70-95 (above average performance)
        self.test_score['good'] = fuzz.trimf(self.test_score.universe, [70, 85, 95])
        # Excellent score: 85-100 (outstanding performance)
        self.test_score['excellent'] = fuzz.trapmf(self.test_score.universe, [85, 95, 100, 100])
        
    def _define_output_variable(self):
        """
        Define the fuzzy output variable for overall performance evaluation.
        
        Output Variable:
        ----------------
        Performance: Range [0, 100]
        - Linguistic terms: low, medium, high
        - Interpretation: Overall academic performance score
          - low: 0-40 (needs improvement)
          - medium: 30-70 (satisfactory)
          - high: 60-100 (excellent)
        """
        self.performance = ctrl.Consequent(np.arange(0, 101, 1), 'performance')
        
        # Low performance: triangular, peaks at 0, decreases to 0 at 40
        self.performance['low'] = fuzz.trimf(self.performance.universe, [0, 0, 40])
        # Medium performance: triangular, centered at 50
        self.performance['medium'] = fuzz.trimf(self.performance.universe, [30, 50, 70])
        # High performance: triangular, starts at 60, peaks at 100
        self.performance['high'] = fuzz.trimf(self.performance.universe, [60, 100, 100])
        
        # Set defuzzification method to centroid (Center of Gravity)
        self.performance.defuzzify_method = 'centroid'
        
    def _define_rules(self):
        """
        Define the fuzzy rule base using linguistic IF-THEN rules.
        
        The rule base encodes expert knowledge about the relationship
        between input variables and student performance. Rules are
        designed to capture realistic academic evaluation criteria.
        
        Rule Base (12 rules):
        ---------------------
        Rules are structured to cover key scenarios in student evaluation,
        considering the interaction between academic metrics.
        """
        
        # Rule 1: Excellent students (high in all positive metrics)
        # IF GPA is high AND test_score is excellent AND participation is active
        # AND absences are few THEN performance is high
        rule1 = ctrl.Rule(
            self.gpa['high'] & self.test_score['excellent'] & 
            self.participation['active'] & self.absences['few'],
            self.performance['high'],
            label='Rule 1: Excellent overall'
        )
        
        # Rule 2: Good students with strong test performance
        # IF GPA is high AND test_score is good THEN performance is high
        rule2 = ctrl.Rule(
            self.gpa['high'] & self.test_score['good'],
            self.performance['high'],
            label='Rule 2: High GPA + Good test'
        )
        
        # Rule 3: Students with high GPA but average test scores
        # IF GPA is high AND test_score is average THEN performance is medium
        rule3 = ctrl.Rule(
            self.gpa['high'] & self.test_score['average'],
            self.performance['medium'],
            label='Rule 3: High GPA + Average test'
        )
        
        # Rule 4: Average students
        # IF GPA is medium AND test_score is average AND participation is moderate
        # THEN performance is medium
        rule4 = ctrl.Rule(
            self.gpa['medium'] & self.test_score['average'] & 
            self.participation['moderate'],
            self.performance['medium'],
            label='Rule 4: Average overall'
        )
        
        # Rule 5: Students with medium GPA but good engagement
        # IF GPA is medium AND participation is active AND absences are few
        # THEN performance is medium
        rule5 = ctrl.Rule(
            self.gpa['medium'] & self.participation['active'] & 
            self.absences['few'],
            self.performance['medium'],
            label='Rule 5: Medium GPA + Active participation'
        )
        
        # Rule 6: Students with low GPA regardless of other factors
        # IF GPA is low AND test_score is poor THEN performance is low
        rule6 = ctrl.Rule(
            self.gpa['low'] & self.test_score['poor'],
            self.performance['low'],
            label='Rule 6: Low GPA + Poor test'
        )
        
        # Rule 7: Poor attendance significantly impacts performance
        # IF absences are many AND participation is passive THEN performance is low
        rule7 = ctrl.Rule(
            self.absences['many'] & self.participation['passive'],
            self.performance['low'],
            label='Rule 7: Poor attendance + Passive'
        )
        
        # Rule 8: Low GPA but showing effort
        # IF GPA is low AND participation is active AND test_score is average
        # THEN performance is medium
        rule8 = ctrl.Rule(
            self.gpa['low'] & self.participation['active'] & 
            self.test_score['average'],
            self.performance['medium'],
            label='Rule 8: Low GPA but active + average test'
        )
        
        # Rule 9: Excellent test compensates for medium GPA
        # IF GPA is medium AND test_score is excellent THEN performance is high
        rule9 = ctrl.Rule(
            self.gpa['medium'] & self.test_score['excellent'],
            self.performance['high'],
            label='Rule 9: Medium GPA + Excellent test'
        )
        
        # Rule 10: Good test with good attendance
        # IF test_score is good AND absences are few AND participation is moderate
        # THEN performance is medium
        rule10 = ctrl.Rule(
            self.test_score['good'] & self.absences['few'] & 
            self.participation['moderate'],
            self.performance['medium'],
            label='Rule 10: Good test + Few absences'
        )
        
        # Rule 11: Poor test score indicates low performance
        # IF test_score is poor AND participation is passive THEN performance is low
        rule11 = ctrl.Rule(
            self.test_score['poor'] & self.participation['passive'],
            self.performance['low'],
            label='Rule 11: Poor test + Passive'
        )
        
        # Rule 12: Medium across the board
        # IF GPA is medium AND test_score is good AND absences are moderate
        # THEN performance is medium
        rule12 = ctrl.Rule(
            self.gpa['medium'] & self.test_score['good'] & 
            self.absences['moderate'],
            self.performance['medium'],
            label='Rule 12: Medium GPA + Good test + Moderate absences'
        )
        
        # Store all rules
        self.rules = [rule1, rule2, rule3, rule4, rule5, rule6, 
                      rule7, rule8, rule9, rule10, rule11, rule12]
        
    def _build_control_system(self):
        """
        Build the fuzzy control system and create the simulation object.
        
        This method constructs the complete Mamdani fuzzy inference system
        using the defined variables and rules. The system uses:
        - AND operation: minimum (min)
        - OR operation: maximum (max)
        - Implication: minimum (Mamdani)
        - Aggregation: maximum
        - Defuzzification: centroid (center of gravity)
        """
        self.system = ctrl.ControlSystem(self.rules)
        self.simulator = ctrl.ControlSystemSimulation(self.system)
        
    def evaluate(self, gpa: float, absences: int, 
                 participation: float, test_score: float) -> float:
        """
        Evaluate student performance using the fuzzy inference system.
        
        Parameters:
        -----------
        gpa : float
            Grade Point Average, range [0, 6]
        absences : int
            Number of absences, range [0, 30]
        participation : float
            Class participation level, range [0, 100]
        test_score : float
            Test score, range [0, 100]
            
        Returns:
        --------
        float
            Defuzzified performance score, range [0, 100]
            
        Raises:
        -------
        ValueError
            If input values are outside their valid ranges
        """
        # Input validation
        if not 2 <= gpa <= 6:
            raise ValueError(f"GPA must be between 2 and 6 (Bulgarian scale), got {gpa}")
        if not 0 <= absences <= 30:
            raise ValueError(f"Absences must be between 0 and 30, got {absences}")
        if not 0 <= participation <= 100:
            raise ValueError(f"Participation must be between 0 and 100, got {participation}")
        if not 0 <= test_score <= 100:
            raise ValueError(f"Test score must be between 0 and 100, got {test_score}")
        
        # Set input values
        self.simulator.input['gpa'] = gpa
        self.simulator.input['absences'] = absences
        self.simulator.input['participation'] = participation
        self.simulator.input['test_score'] = test_score
        
        # Perform fuzzy inference (compositional rule of inference)
        self.simulator.compute()
        
        # Return defuzzified output
        return self.simulator.output['performance']
    
    def get_performance_category(self, score: float) -> str:
        """
        Convert numerical performance score to linguistic category.
        
        Parameters:
        -----------
        score : float
            Performance score [0, 100]
            
        Returns:
        --------
        str
            Linguistic category: 'Low', 'Medium', or 'High'
        """
        if score < 35:
            return "Low"
        elif score < 65:
            return "Medium"
        else:
            return "High"
    
    def evaluate_detailed(self, gpa: float, absences: int,
                          participation: float, test_score: float) -> Dict:
        """
        Perform detailed evaluation with membership degree information.
        
        Parameters:
        -----------
        gpa : float
            Grade Point Average, range [0, 6]
        absences : int
            Number of absences, range [0, 30]
        participation : float
            Class participation level, range [0, 100]
        test_score : float
            Test score, range [0, 100]
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'performance_score': Defuzzified performance value
            - 'category': Linguistic performance category
            - 'input_memberships': Membership degrees for each input
            - 'inputs': Original input values
        """
        # Calculate performance score
        score = self.evaluate(gpa, absences, participation, test_score)
        
        # Calculate membership degrees for inputs
        gpa_memberships = {
            'low': fuzz.interp_membership(self.gpa.universe, 
                                          self.gpa['low'].mf, gpa),
            'medium': fuzz.interp_membership(self.gpa.universe, 
                                             self.gpa['medium'].mf, gpa),
            'high': fuzz.interp_membership(self.gpa.universe, 
                                           self.gpa['high'].mf, gpa)
        }
        
        absence_memberships = {
            'few': fuzz.interp_membership(self.absences.universe, 
                                          self.absences['few'].mf, absences),
            'moderate': fuzz.interp_membership(self.absences.universe, 
                                               self.absences['moderate'].mf, absences),
            'many': fuzz.interp_membership(self.absences.universe, 
                                           self.absences['many'].mf, absences)
        }
        
        participation_memberships = {
            'passive': fuzz.interp_membership(self.participation.universe, 
                                              self.participation['passive'].mf, participation),
            'moderate': fuzz.interp_membership(self.participation.universe, 
                                               self.participation['moderate'].mf, participation),
            'active': fuzz.interp_membership(self.participation.universe, 
                                             self.participation['active'].mf, participation)
        }
        
        test_memberships = {
            'poor': fuzz.interp_membership(self.test_score.universe, 
                                           self.test_score['poor'].mf, test_score),
            'average': fuzz.interp_membership(self.test_score.universe, 
                                              self.test_score['average'].mf, test_score),
            'good': fuzz.interp_membership(self.test_score.universe, 
                                           self.test_score['good'].mf, test_score),
            'excellent': fuzz.interp_membership(self.test_score.universe, 
                                                self.test_score['excellent'].mf, test_score)
        }
        
        return {
            'performance_score': score,
            'category': self.get_performance_category(score),
            'input_memberships': {
                'gpa': gpa_memberships,
                'absences': absence_memberships,
                'participation': participation_memberships,
                'test_score': test_memberships
            },
            'inputs': {
                'gpa': gpa,
                'absences': absences,
                'participation': participation,
                'test_score': test_score
            }
        }


# =============================================================================
# SECTION 2: VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_membership_functions(evaluator: FuzzyStudentEvaluator, 
                                   save_path: str = None):
    """
    Visualize all membership functions for input and output variables.
    
    Creates a comprehensive figure showing the membership functions for:
    - GPA (Grade Point Average)
    - Absences (Number of missed classes)
    - Participation (Class engagement level)
    - Test Score (Examination performance)
    - Performance (Output variable)
    
    Parameters:
    -----------
    evaluator : FuzzyStudentEvaluator
        The fuzzy expert system instance
    save_path : str, optional
        Path to save the figure. If None, displays interactively.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Membership Functions for Student Performance Evaluation System',
                 fontsize=14, fontweight='bold')
    
    # Color scheme for consistency
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    
    # Plot 1: GPA membership functions
    ax1 = axes[0, 0]
    ax1.plot(evaluator.gpa.universe, evaluator.gpa['low'].mf, 
             colors[0], linewidth=2, label='Low')
    ax1.plot(evaluator.gpa.universe, evaluator.gpa['medium'].mf, 
             colors[1], linewidth=2, label='Medium')
    ax1.plot(evaluator.gpa.universe, evaluator.gpa['high'].mf, 
             colors[2], linewidth=2, label='High')
    ax1.set_title('GPA (Grade Point Average)', fontweight='bold')
    ax1.set_xlabel('GPA Value')
    ax1.set_ylabel('Membership Degree μ(x)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 6])
    ax1.set_ylim([0, 1.05])
    
    # Plot 2: Absences membership functions
    ax2 = axes[0, 1]
    ax2.plot(evaluator.absences.universe, evaluator.absences['few'].mf, 
             colors[2], linewidth=2, label='Few')
    ax2.plot(evaluator.absences.universe, evaluator.absences['moderate'].mf, 
             colors[1], linewidth=2, label='Moderate')
    ax2.plot(evaluator.absences.universe, evaluator.absences['many'].mf, 
             colors[0], linewidth=2, label='Many')
    ax2.set_title('Number of Absences', fontweight='bold')
    ax2.set_xlabel('Number of Absences')
    ax2.set_ylabel('Membership Degree μ(x)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 30])
    ax2.set_ylim([0, 1.05])
    
    # Plot 3: Participation membership functions
    ax3 = axes[1, 0]
    ax3.plot(evaluator.participation.universe, evaluator.participation['passive'].mf, 
             colors[0], linewidth=2, label='Passive')
    ax3.plot(evaluator.participation.universe, evaluator.participation['moderate'].mf, 
             colors[1], linewidth=2, label='Moderate')
    ax3.plot(evaluator.participation.universe, evaluator.participation['active'].mf, 
             colors[2], linewidth=2, label='Active')
    ax3.set_title('Class Participation Level', fontweight='bold')
    ax3.set_xlabel('Participation Score (%)')
    ax3.set_ylabel('Membership Degree μ(x)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 100])
    ax3.set_ylim([0, 1.05])
    
    # Plot 4: Test Score membership functions
    ax4 = axes[1, 1]
    ax4.plot(evaluator.test_score.universe, evaluator.test_score['poor'].mf, 
             colors[0], linewidth=2, label='Poor')
    ax4.plot(evaluator.test_score.universe, evaluator.test_score['average'].mf, 
             colors[1], linewidth=2, label='Average')
    ax4.plot(evaluator.test_score.universe, evaluator.test_score['good'].mf, 
             colors[2], linewidth=2, label='Good')
    ax4.plot(evaluator.test_score.universe, evaluator.test_score['excellent'].mf, 
             colors[3], linewidth=2, label='Excellent')
    ax4.set_title('Test Score', fontweight='bold')
    ax4.set_xlabel('Test Score (%)')
    ax4.set_ylabel('Membership Degree μ(x)')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 100])
    ax4.set_ylim([0, 1.05])
    
    # Plot 5: Performance (Output) membership functions
    ax5 = axes[2, 0]
    ax5.plot(evaluator.performance.universe, evaluator.performance['low'].mf, 
             colors[0], linewidth=2, label='Low')
    ax5.plot(evaluator.performance.universe, evaluator.performance['medium'].mf, 
             colors[1], linewidth=2, label='Medium')
    ax5.plot(evaluator.performance.universe, evaluator.performance['high'].mf, 
             colors[2], linewidth=2, label='High')
    ax5.set_title('Overall Performance (Output)', fontweight='bold')
    ax5.set_xlabel('Performance Score (%)')
    ax5.set_ylabel('Membership Degree μ(x)')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, 100])
    ax5.set_ylim([0, 1.05])
    
    # Hide the unused subplot
    axes[2, 1].axis('off')
    
    # Add legend explanation
    axes[2, 1].text(0.5, 0.7, 'Membership Function Types:', 
                    transform=axes[2, 1].transAxes, fontsize=11, 
                    fontweight='bold', ha='center')
    axes[2, 1].text(0.5, 0.5, '• Triangular (trimf): GPA, Absences,\n  Participation, some Test Score terms',
                    transform=axes[2, 1].transAxes, fontsize=10, ha='center')
    axes[2, 1].text(0.5, 0.3, '• Trapezoidal (trapmf): Test Score\n  (poor, excellent), Performance',
                    transform=axes[2, 1].transAxes, fontsize=10, ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    return fig


def visualize_evaluation_result(evaluator: FuzzyStudentEvaluator,
                                gpa: float, absences: int,
                                participation: float, test_score: float,
                                save_path: str = None):
    """
    Visualize the evaluation result for a specific student.
    
    Creates a figure showing:
    - Input values marked on membership functions
    - Output aggregation and defuzzification result
    
    Parameters:
    -----------
    evaluator : FuzzyStudentEvaluator
        The fuzzy expert system instance
    gpa, absences, participation, test_score : numeric
        Input values for evaluation
    save_path : str, optional
        Path to save the figure
    """
    # Get detailed evaluation
    result = evaluator.evaluate_detailed(gpa, absences, participation, test_score)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Evaluation Result: Performance = {result["performance_score"]:.2f} ({result["category"]})',
                 fontsize=14, fontweight='bold')
    
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    
    # Plot GPA with input marker
    ax1 = axes[0, 0]
    ax1.plot(evaluator.gpa.universe, evaluator.gpa['low'].mf, colors[0], linewidth=2, label='Low')
    ax1.plot(evaluator.gpa.universe, evaluator.gpa['medium'].mf, colors[1], linewidth=2, label='Medium')
    ax1.plot(evaluator.gpa.universe, evaluator.gpa['high'].mf, colors[2], linewidth=2, label='High')
    ax1.axvline(x=gpa, color='black', linestyle='--', linewidth=2, label=f'Input: {gpa}')
    ax1.set_title(f'GPA = {gpa}', fontweight='bold')
    ax1.set_xlabel('GPA')
    ax1.set_ylabel('μ(x)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot Absences with input marker
    ax2 = axes[0, 1]
    ax2.plot(evaluator.absences.universe, evaluator.absences['few'].mf, colors[2], linewidth=2, label='Few')
    ax2.plot(evaluator.absences.universe, evaluator.absences['moderate'].mf, colors[1], linewidth=2, label='Moderate')
    ax2.plot(evaluator.absences.universe, evaluator.absences['many'].mf, colors[0], linewidth=2, label='Many')
    ax2.axvline(x=absences, color='black', linestyle='--', linewidth=2, label=f'Input: {absences}')
    ax2.set_title(f'Absences = {absences}', fontweight='bold')
    ax2.set_xlabel('Absences')
    ax2.set_ylabel('μ(x)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot Participation with input marker
    ax3 = axes[0, 2]
    ax3.plot(evaluator.participation.universe, evaluator.participation['passive'].mf, colors[0], linewidth=2, label='Passive')
    ax3.plot(evaluator.participation.universe, evaluator.participation['moderate'].mf, colors[1], linewidth=2, label='Moderate')
    ax3.plot(evaluator.participation.universe, evaluator.participation['active'].mf, colors[2], linewidth=2, label='Active')
    ax3.axvline(x=participation, color='black', linestyle='--', linewidth=2, label=f'Input: {participation}')
    ax3.set_title(f'Participation = {participation}', fontweight='bold')
    ax3.set_xlabel('Participation (%)')
    ax3.set_ylabel('μ(x)')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot Test Score with input marker
    ax4 = axes[1, 0]
    ax4.plot(evaluator.test_score.universe, evaluator.test_score['poor'].mf, colors[0], linewidth=2, label='Poor')
    ax4.plot(evaluator.test_score.universe, evaluator.test_score['average'].mf, colors[1], linewidth=2, label='Average')
    ax4.plot(evaluator.test_score.universe, evaluator.test_score['good'].mf, colors[2], linewidth=2, label='Good')
    ax4.plot(evaluator.test_score.universe, evaluator.test_score['excellent'].mf, colors[3], linewidth=2, label='Excellent')
    ax4.axvline(x=test_score, color='black', linestyle='--', linewidth=2, label=f'Input: {test_score}')
    ax4.set_title(f'Test Score = {test_score}', fontweight='bold')
    ax4.set_xlabel('Test Score (%)')
    ax4.set_ylabel('μ(x)')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Plot Output with result marker
    ax5 = axes[1, 1]
    ax5.plot(evaluator.performance.universe, evaluator.performance['low'].mf, colors[0], linewidth=2, label='Low')
    ax5.plot(evaluator.performance.universe, evaluator.performance['medium'].mf, colors[1], linewidth=2, label='Medium')
    ax5.plot(evaluator.performance.universe, evaluator.performance['high'].mf, colors[2], linewidth=2, label='High')
    ax5.axvline(x=result['performance_score'], color='black', linestyle='--', linewidth=2, 
                label=f'Output: {result["performance_score"]:.2f}')
    ax5.fill_between(evaluator.performance.universe, 0, evaluator.performance['medium'].mf, 
                     alpha=0.3, color=colors[1])
    ax5.set_title(f'Performance = {result["performance_score"]:.2f}', fontweight='bold')
    ax5.set_xlabel('Performance (%)')
    ax5.set_ylabel('μ(x)')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Membership degrees summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = "Input Membership Degrees:\n\n"
    for var_name, memberships in result['input_memberships'].items():
        summary_text += f"{var_name.upper()}:\n"
        for term, degree in memberships.items():
            if degree > 0.001:
                summary_text += f"  • {term}: {degree:.3f}\n"
        summary_text += "\n"
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    return fig


def visualize_system_response_surface(evaluator: FuzzyStudentEvaluator,
                                      fixed_participation: float = 50,
                                      fixed_absences: int = 10,
                                      save_path: str = None):
    """
    Visualize the system response surface for GPA vs Test Score.
    
    Creates a 3D surface plot showing how performance varies with
    GPA and Test Score while holding other variables constant.
    
    Parameters:
    -----------
    evaluator : FuzzyStudentEvaluator
        The fuzzy expert system instance
    fixed_participation : float
        Fixed participation value for the surface
    fixed_absences : int
        Fixed absences value for the surface
    save_path : str, optional
        Path to save the figure
    """
    # Create meshgrid for GPA and Test Score
    gpa_range = np.linspace(0, 6, 25)
    test_range = np.linspace(0, 100, 25)
    GPA, TEST = np.meshgrid(gpa_range, test_range)
    
    # Calculate performance for each combination
    PERF = np.zeros_like(GPA)
    for i in range(GPA.shape[0]):
        for j in range(GPA.shape[1]):
            try:
                PERF[i, j] = evaluator.evaluate(
                    GPA[i, j], fixed_absences, fixed_participation, TEST[i, j]
                )
            except:
                PERF[i, j] = np.nan
    
    # Create figure
    fig = plt.figure(figsize=(12, 5))
    
    # 3D Surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(GPA, TEST, PERF, cmap='viridis', alpha=0.8,
                            linewidth=0, antialiased=True)
    ax1.set_xlabel('GPA')
    ax1.set_ylabel('Test Score')
    ax1.set_zlabel('Performance')
    ax1.set_title(f'Response Surface\n(Participation={fixed_participation}, Absences={fixed_absences})')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Performance')
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(GPA, TEST, PERF, levels=20, cmap='viridis')
    ax2.set_xlabel('GPA')
    ax2.set_ylabel('Test Score')
    ax2.set_title(f'Performance Contour Map\n(Participation={fixed_participation}, Absences={fixed_absences})')
    fig.colorbar(contour, ax=ax2, label='Performance')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    return fig


# =============================================================================
# SECTION 3: DEMONSTRATION AND TESTING
# =============================================================================

def run_demonstration():
    """
    Run a complete demonstration of the fuzzy expert system.
    
    This function:
    1. Creates the fuzzy expert system
    2. Displays all membership functions
    3. Evaluates multiple test cases
    4. Visualizes results
    """
    print("=" * 70)
    print("FUZZY EXPERT SYSTEM FOR STUDENT ACADEMIC PERFORMANCE EVALUATION")
    print("=" * 70)
    print()
    
    # Create the fuzzy expert system
    print("Initializing Fuzzy Expert System...")
    evaluator = FuzzyStudentEvaluator()
    print("System initialized successfully!")
    print()
    
    # Display rule base
    print("-" * 70)
    print("FUZZY RULE BASE (12 Rules)")
    print("-" * 70)
    for i, rule in enumerate(evaluator.rules, 1):
        print(f"Rule {i}: {rule.label}")
    print()
    
    # Define test cases representing different student profiles
    test_cases = [
        {
            'name': 'Excellent Student',
            'gpa': 5.5, 'absences': 2, 'participation': 90, 'test_score': 95,
            'description': 'High achiever in all aspects'
        },
        {
            'name': 'Average Student',
            'gpa': 3.5, 'absences': 10, 'participation': 50, 'test_score': 60,
            'description': 'Balanced performance across metrics'
        },
        {
            'name': 'Struggling Student',
            'gpa': 1.5, 'absences': 20, 'participation': 25, 'test_score': 35,
            'description': 'Needs significant improvement'
        },
        {
            'name': 'Test Ace, Low GPA',
            'gpa': 2.0, 'absences': 5, 'participation': 60, 'test_score': 88,
            'description': 'Excellent test performance despite lower GPA'
        },
        {
            'name': 'Active but Struggling',
            'gpa': 2.5, 'absences': 3, 'participation': 85, 'test_score': 55,
            'description': 'High engagement but difficulty with material'
        },
        {
            'name': 'Absent High Achiever',
            'gpa': 4.5, 'absences': 18, 'participation': 30, 'test_score': 75,
            'description': 'Good grades despite poor attendance'
        },
        {
            'name': 'Consistent Medium',
            'gpa': 3.0, 'absences': 8, 'participation': 55, 'test_score': 65,
            'description': 'Steady medium performance'
        },
        {
            'name': 'Perfect Attendance, Low Scores',
            'gpa': 2.0, 'absences': 0, 'participation': 70, 'test_score': 45,
            'description': 'Present and engaged but struggling academically'
        }
    ]
    
    # Evaluate all test cases
    print("-" * 70)
    print("EVALUATION RESULTS")
    print("-" * 70)
    print()
    
    results = []
    for case in test_cases:
        result = evaluator.evaluate_detailed(
            case['gpa'], case['absences'], 
            case['participation'], case['test_score']
        )
        results.append(result)
        
        print(f"Student Profile: {case['name']}")
        print(f"Description: {case['description']}")
        print(f"  Inputs:")
        print(f"    - GPA: {case['gpa']}")
        print(f"    - Absences: {case['absences']}")
        print(f"    - Participation: {case['participation']}%")
        print(f"    - Test Score: {case['test_score']}%")
        print(f"  Output:")
        print(f"    - Performance Score: {result['performance_score']:.2f}")
        print(f"    - Category: {result['category']}")
        print()
    
    # Summary statistics
    print("-" * 70)
    print("SUMMARY STATISTICS")
    print("-" * 70)
    scores = [r['performance_score'] for r in results]
    print(f"  Minimum Performance: {min(scores):.2f}")
    print(f"  Maximum Performance: {max(scores):.2f}")
    print(f"  Average Performance: {np.mean(scores):.2f}")
    print(f"  Standard Deviation: {np.std(scores):.2f}")
    print()
    
    # Generate visualizations
    print("-" * 70)
    print("GENERATING VISUALIZATIONS")
    print("-" * 70)
    
    # Save membership functions plot
    print("1. Generating membership functions visualization...")
    visualize_membership_functions(evaluator, 'membership_functions.png')
    
    # Save evaluation result for excellent student
    print("2. Generating evaluation result for 'Excellent Student'...")
    visualize_evaluation_result(evaluator, 5.5, 2, 90, 95, 'evaluation_excellent.png')
    
    # Save evaluation result for struggling student
    print("3. Generating evaluation result for 'Struggling Student'...")
    visualize_evaluation_result(evaluator, 1.5, 20, 25, 35, 'evaluation_struggling.png')
    
    # Save response surface
    print("4. Generating system response surface...")
    visualize_system_response_surface(evaluator, save_path='response_surface.png')
    
    print()
    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("Generated files:")
    print("  - membership_functions.png")
    print("  - evaluation_excellent.png")
    print("  - evaluation_struggling.png")
    print("  - response_surface.png")
    
    return evaluator, results


def interactive_evaluation():
    """
    Run an interactive evaluation session.
    
    Allows users to input custom values and see results.
    """
    print("\n" + "=" * 70)
    print("INTERACTIVE STUDENT EVALUATION")
    print("=" * 70)
    
    evaluator = FuzzyStudentEvaluator()
    
    while True:
        print("\nEnter student data (or 'quit' to exit):")
        
        try:
            gpa_input = input("  GPA (0-6): ")
            if gpa_input.lower() == 'quit':
                break
            gpa = float(gpa_input)
            
            absences = int(input("  Absences (0-30): "))
            participation = float(input("  Participation (0-100): "))
            test_score = float(input("  Test Score (0-100): "))
            
            result = evaluator.evaluate_detailed(gpa, absences, participation, test_score)
            
            print(f"\n  >>> Performance Score: {result['performance_score']:.2f}")
            print(f"  >>> Category: {result['category']}")
            
        except ValueError as e:
            print(f"  Error: {e}")
        except KeyboardInterrupt:
            break
    
    print("\nExiting interactive evaluation.")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run the demonstration
    evaluator, results = run_demonstration()
    
    # Uncomment the following line for interactive mode:
    # interactive_evaluation()
