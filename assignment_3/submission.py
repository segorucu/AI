import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy
import pgmpy
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    BayesNet = BayesianModel(
        [("temperature", "faulty gauge"), ("temperature", "gauge"), ("faulty gauge", "gauge"), ("gauge", "alarm"),
         ("faulty alarm", "alarm")])

    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    # TODO: set the probability distribution for each node
    cpd_T = TabularCPD("temperature", 2, values=[[0.8], [0.2]])
    cpd_FA = TabularCPD("faulty alarm", 2, values=[[0.85], [0.15]])

    cpd_FG = TabularCPD('faulty gauge', 2, values=[[0.95, 0.2], \
                                                   [0.05, 0.8]], evidence=['temperature'], evidence_card=[2])

    cpd_G = TabularCPD('gauge', 2, values=[[0.95, 0.2, 0.05, 0.8], \
                                           [0.05, 0.8, 0.95, 0.2]], \
                       evidence=['temperature', 'faulty gauge'], evidence_card=[2, 2])

    cpd_A = TabularCPD('alarm', 2, values=[[0.9, 0.55, 0.1, 0.45], \
                                           [0.1, 0.45, 0.9, 0.55]], evidence=['gauge', 'faulty alarm'], \
                       evidence_card=[2, 2])
    bayes_net.add_cpds(cpd_T, cpd_FA, cpd_FG, cpd_G, cpd_A)

    # alarm_p = get_alarm_prob(bayes_net)
    # gauge_p = get_gauge_prob(bayes_net)
    # temp_p = get_temperature_prob(bayes_net)

    return bayes_net


def get_alarm_prob(bayes_net):
    """Calculate the marginal 
    probability of the alarm 
    ringing in the 
    power plant system."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['alarm'], joint=False)
    alarm_prob = marginal_prob['alarm'].values[1]
    return alarm_prob


def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge 
    showing hot in the 
    power plant system."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['gauge'], joint=False)
    gauge_prob = marginal_prob['gauge'].values[1]
    return gauge_prob


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability 
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['temperature'],
                                    evidence={'alarm': 1, 'faulty alarm': 0, 'faulty gauge': 0}, joint=False)
    temp_prob = conditional_prob['temperature'].values[1]
    return temp_prob


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel([("A", "AvB"), ("B", "AvB"),
                              ("A", "CvA"), ("C", "CvA"),
                              ("B", "BvC"), ("C", "BvC")])

    val0 = [[0.15], [0.45], [0.3], [0.1]]
    cpd_A = TabularCPD("A", 4, values=val0)
    cpd_B = TabularCPD("B", 4, values=val0)
    cpd_C = TabularCPD("C", 4, values=val0)

    val1 = [[0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1],
            [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1],
            [0.8, 0.2, 0.1, 0.05, 0.2, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2, 0.05, 0.1, 0.2, 0.8]]

    cpd_AvB = TabularCPD('AvB', 3, values=val1,
                         evidence=["A", "B"], evidence_card=[4, 4])
    cpd_CvA = TabularCPD('CvA', 3, values=val1,
                         evidence=["C", "A"], evidence_card=[4, 4])
    cpd_BvC = TabularCPD('BvC', 3, values=val1,
                         evidence=["B", "C"], evidence_card=[4, 4])

    BayesNet.add_cpds(cpd_A, cpd_B, cpd_C, cpd_AvB, cpd_CvA, cpd_BvC)

    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0, 0, 0]
    # TODO: finish this function    
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['BvC'], evidence={'AvB': 0, 'CvA': 2}, joint=False)
    posterior = conditional_prob['BvC'].values
    return posterior  # list


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    team_table = bayes_net.get_cpds("A").values
    match_table = bayes_net.get_cpds("AvB").values

    # (('A', 'AvB', 'B', 'CvA', 'C', 'BvC'))
    if initial_state is None or len(initial_state) != 6:
        options = [0, 1, 2, 3]
        A, B, C = np.random.choice(options, 3)[0:3]
        options = [0, 1, 2]
        BvC = np.random.choice(options, 1)[0]
        posterior_state = tuple([A, B, C, 0, BvC, 2])
    else:
        hidden = ['A', 'B', 'C', 'BvC']
        sel = np.random.choice(hidden, 1)[0]
        p = np.random.random()
        if sel == 'A' or sel == 'B' or sel == 'C':
            A, B, C, AvB, BvC, CvA = initial_state
            if sel == 'A':
                v1 = match_table[0, :, B]
                v2 = match_table[2, C, :]
                denom = 0.
                valup = []
                for i in range(4):
                    valup.append(v1[i] * v2[i] * team_table[i])
                    denom += valup[i]
                for i in range(4):
                    valup[i] /= denom
            elif sel == 'B':
                v1 = match_table[0, A, :]
                v2 = match_table[BvC, :, C]
                denom = 0.
                valup = []
                for i in range(4):
                    valup.append(v1[i] * v2[i] * team_table[i])
                    denom += valup[i]
                for i in range(4):
                    valup[i] /= denom
            elif sel == 'C':
                v1 = match_table[BvC, B, :]
                v2 = match_table[2, :, A]
                denom = 0.
                valup = []
                for i in range(4):
                    valup.append(v1[i] * v2[i] * team_table[i])
                    denom += valup[i]
                for i in range(4):
                    valup[i] /= denom
            for i in range(len(valup)):
                val = sum(valup[0:i + 1])
                if p < val:
                    ival = i
                    break
        else:
            col = match_table[:, initial_state[1], initial_state[2]]
            for i in range(len(col)):
                val = sum(col[0:i + 1])
                if p < val:
                    ival = i
                    break
        nodes = ['A', 'B', 'C', 'AvB', 'BvC', 'CvA']
        ind = nodes.index(sel)
        posterior_state = initial_state
        posterior_state[ind] = ival
        posterior_state = tuple(posterior_state)


    return posterior_state


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A_cpd = bayes_net.get_cpds("A")
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values
    # TODO: finish this function
    if initial_state is None or len(initial_state) != 6:
        options = [0, 1, 2, 3]
        A, B, C = np.random.choice(options, 3)[0:3]
        options = [0, 1, 2]
        BvC = np.random.choice(options, 1)[0]
        posterior_state = tuple([A, B, C, 0, BvC, 2])
    else:
        prob = calc_prob(team_table, match_table, initial_state)

        options = [0, 1, 2, 3]
        A, B, C = np.random.choice(options, 3)[0:3]
        options = [0, 1, 2]
        BvC = np.random.choice(options, 1)[0]
        candidate = [A, B, C, 0, BvC, 2]

        probn = calc_prob(team_table, match_table, candidate)
        alpha = probn / prob

        p = np.random.random()
        if alpha > p:
            posterior_state = candidate
        else:
            posterior_state = initial_state
        posterior_state = tuple(posterior_state)

    return posterior_state

def calc_prob(team_table, match_table, state):

    A, B, C, AvB, BvC, CvA = state
    pA = team_table[A]
    pB = team_table[B]
    pC = team_table[C]
    pAvB = match_table[0, A, B]
    pCvA = match_table[2, C, A]
    pBvC = match_table[BvC, B, C]
    probability = pA * pB * pC * pAvB * pCvA * pBvC

    return probability

def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0, 0, 0]  # posterior distribution of the BvC match as produced by Gibbs
    MH_convergence = [0, 0, 0]  # posterior distribution of the BvC match as produced by MH
    delta = 0.0001
    N = 100
    burn_in = 5000
    # TODO: finish this function
    goal = [0.25890074, 0.42796763, 0.31313163]
    total = []
    k = 0
    before = [0, 0, 0]
    initial_state = []
    for i in range(1000000):
        posterior = Gibbs_sampler(bayes_net, initial_state)
        initial_state = list(posterior)
        total.append(posterior)
        if i > burn_in:
            v0, v1, v2 = 0, 0, 0
            tot = len(total)
            for row in total:
                BvC = row[4]
                if BvC == 0:
                    v0 += 1
                elif BvC == 1:
                    v1 += 1
                elif BvC == 2:
                    v2 += 1
            now = [v0 / tot, v1 / tot, v2 / tot]
            mv = 0.
            for j in range(3):
                if abs(now[j]-before[j]) > mv:
                    mv = abs(now[j]-before[j])
            before = now
            if mv < delta:
                k += 1
            else:
                k = 0
            if k > N:
                Gibbs_convergence = now
                Gibbs_count = i
                print(Gibbs_count,Gibbs_convergence)
                break
    # 0.25890074, 0.42796763, 0.31313163]

    total = []
    initial_state = []
    for i in range(1000000):
        posterior = MH_sampler(bayes_net, initial_state)
        if list(posterior) == initial_state:
            MH_rejection_count += 1
        initial_state = list(posterior)
        total.append(posterior)

        if i > burn_in * 2:
            v0, v1, v2 = 0, 0, 0
            tot = len(total)
            for row in total:
                BvC = row[4]
                if BvC == 0:
                    v0 += 1
                elif BvC == 1:
                    v1 += 1
                elif BvC == 2:
                    v2 += 1
            now = [v0 / tot, v1 / tot, v2 / tot]
            mv = 0.
            for j in range(3):
                if abs(now[j] - before[j]) > mv:
                    mv = abs(now[j] - before[j])
            before = now
            if mv < delta * 0.5:
                k += 1
            else:
                k = 0
            if k > N:
                MH_convergence = now
                MH_count = i
                print(MH_count, MH_convergence, MH_rejection_count)
                break

    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor

    choice = 1
    options = ['Gibbs', 'Metropolis-Hastings']
    factor = 2
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "sgorucu3"
