# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

import heapq
import os
import pickle
import math
from copy import deepcopy
import itertools
import pandas as pd


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []
        self.counter = itertools.count()

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        # TODO: finish this function!

        node = heapq.heappop(self.queue)
        return node

    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """

        self.queue.remove(node)

        return

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        # TODO: finish this function!
        count = next(self.counter)
        if len(node) == 2:
            heap = (node[0], count, node[1])
        elif len(node) == 3:
            heap = (node[0], count, node[1], node[2])
        heapq.heappush(self.queue, heap)
        return

    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in the queue.
        """

        return self.queue[0]


def findpath(policy, start, goal):
    path = [goal]

    child = goal
    while child != start:
        for key, value in policy.items():
            if child in value:
                path.append(key)
                child = key
                break
    path.reverse()

    return path


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    path = []
    if start == goal:
        return path
    frontier = [start]
    explored = []
    queue = PriorityQueue()
    queue.append((0, start))
    policy = {}
    while (1):
        if len(queue.queue) == 0:
            path = findpath(policy, start, goal)
            return path
        (cost, _, node) = queue.pop()
        explored.append(node)
        frontier.remove(node)
        neighbors = sorted(list(graph.neighbors(node)))
        for child in neighbors:
            if child in explored or child in frontier:
                pass
            else:
                frontier.append(child)
                queue.append((cost + 1, child))
                policyappend(policy, node, child)

                if child == goal:
                    path = findpath(policy, start, goal)
                    return path

    return


def policyappend(policy, node, child):
    try:
        policy[node].append(child)
    except:
        policy[node] = [child]

    return


def removechild(policy, child):
    for key, value in policy.items():
        if child in value:
            policy[key].remove(child)

    return


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    path = []
    if start == goal:
        return path
    frontier = [start]
    explored = []
    queue = PriorityQueue()
    queue.append((0, start))
    min_cost = 1e+20
    policy = {}
    while (1):
        if len(queue.queue) == 0:
            path_sel = findpath(policy, start, goal)
            return path_sel
        (cost, _, node) = queue.pop()
        if node == goal:
            continue
        explored.append(node)
        if node in frontier:
            frontier.remove(node)
        if cost >= min_cost:
            continue
        neighbors = sorted(list(graph.neighbors(node)))
        for child in neighbors:
            u_c = graph.get_edge_weight(node, child)
            if child in explored:
                continue
            elif child in frontier:
                for i in range(len(queue.queue)):
                    if queue.queue[i][2] == child:
                        if cost + u_c < queue.queue[i][0]:
                            queue.remove(queue.queue[i])
                            queue.append((cost + u_c, child))
                            removechild(policy, child)
                            policyappend(policy, node, child)
            if child == goal and cost + u_c < min_cost:
                removechild(policy, child)
                policyappend(policy, node, child)
                min_cost = cost + u_c
            elif child != goal and child not in frontier:
                frontier.append(child)
                queue.append((cost + u_c, child))
                policyappend(policy, node, child)

    return path_sel


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """
    a = graph.nodes[v]['pos']
    b = graph.nodes[goal]['pos']
    dist = (a[0] - b[0]) ** 2.0 + (a[1] - b[1]) ** 2.0
    dist = dist ** 0.5
    # TODO: finish this function!
    return dist


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!

    path = []
    if start == goal:
        return path
    frontier = [start]
    explored = []
    queue = PriorityQueue()
    #             f  g  child
    queue.append((0, 0, start))
    min_cost = 1e+20
    policy = {}
    while (1):
        if len(queue.queue) == 0:
            path_sel = findpath(policy, start, goal)
            return path_sel
        (fval, _, cost, node) = queue.pop()
        if node == goal:
            continue
        explored.append(node)
        if node in frontier:
            frontier.remove(node)
        if fval >= min_cost:
            continue
        neighbors = graph[node]
        for child, val in sorted(neighbors.items()):
            u_c = val['weight']
            hval = euclidean_dist_heuristic(graph, child, goal)
            fval = cost + u_c + hval
            if child in explored:
                continue
            elif child in frontier:
                for i in range(len(queue.queue)):
                    if queue.queue[i][3] == child:
                        if fval < queue.queue[i][0]:
                            queue.remove(queue.queue[i])
                            queue.append((fval, cost + u_c, child))
                            removechild(policy, child)
                            policyappend(policy, node, child)
            if child == goal and fval < min_cost:
                removechild(policy, child)
                policyappend(policy, node, child)
                min_cost = fval
            elif child != goal and child not in frontier:
                frontier.append(child)
                queue.append((fval, cost + u_c, child))
                policyappend(policy, node, child)

    return path_sel


def bi_ucs_step(graph, node, path, explored, frontier, cost, queue, policy_s, policy_g, mode):
    new_frontier = []
    updated_frontier = []
    neighbors = graph[node]
    for child, val in sorted(neighbors.items()):
        u_c = val['weight']
        if child in explored:
            continue
        elif child in frontier:
            for i in range(len(queue.queue)):
                if queue.queue[i][2] == child:
                    if cost + u_c < queue.queue[i][0]:
                        queue.remove(queue.queue[i])
                        queue.append((cost + u_c, child))
                        if mode == 'f':
                            removechild(policy_s, child)
                            policyappend(policy_s, node, child)
                        elif mode == 'b':
                            removechild(policy_g, child)
                            policyappend(policy_g, node, child)
                        updated_frontier.append(child)
        else:
            frontier.append(child)
            queue.append((cost + u_c, child))
            new_frontier.append(child)
            if mode == 'f':
                policyappend(policy_s, node, child)
            elif mode == 'b':
                policyappend(policy_g, node, child)

    return queue, new_frontier, updated_frontier


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    # TODO: finish this function!

    path = []
    if start == goal:
        return path
    frontier_s = [start]
    explored = []
    frontier_g = [goal]
    queue_s = PriorityQueue()
    queue_s.append((0, start))
    queue_g = PriorityQueue()
    queue_g.append((0, goal))
    min_cost = 1e+20
    min_cost_s = 1e+20
    min_cost_g = 1e+20
    policy_s = {}
    policy_g = {}
    communicated = {"node": [], "cost_s": [], "cost_g": [], "min_cost": []}
    while (1):
        if len(queue_s.queue) == 0 and len(queue_g.queue) == 0:
            break
        fcost = 0.
        if len(queue_s.queue) > 0:
            fcost = queue_s.queue[0][0]
        bcost = 0
        if len(queue_g.queue) > 0:
            bcost = queue_g.queue[0][0]
        if fcost >= min_cost_s and bcost >= min_cost_g or\
                len(queue_s.queue) == 0 and bcost >= min_cost_g or\
                len(queue_g.queue) == 0 and fcost >= min_cost_s:
            break

        if len(queue_s.queue) > 0:
            (cost, _, node) = queue_s.pop()
            if node in communicated["node"] or cost >= min_cost_s:
                continue
            explored.append(node)
            if node in frontier_s:
                frontier_s.remove(node)

            queue_s, new_frontier, updated_frontier = bi_ucs_step(graph, node, path, explored, frontier_s, cost, queue_s, policy_s, policy_g, 'f')
            for front in updated_frontier:
                if front in communicated["node"]:
                    for i in range(len(queue_s.queue)):
                        if front == queue_s.queue[i][2]:
                            break
                    k = communicated["node"].index(front)
                    if queue_s.queue[i][0] < communicated["cost_s"][k]:
                        communicated["cost_s"][k] = queue_s.queue[i][0]
                        communicated["min_cost"][k] = queue_s.queue[i][0] + communicated["cost_g"][k]
                        communicated["node"][k] = front
                    if communicated["min_cost"][k] < min_cost:
                        min_cost = communicated["min_cost"][k]
                        min_front = front
                        min_cost_s = communicated["cost_s"][k]
                        min_cost_g = communicated["cost_g"][k]

            for front in new_frontier:
                if front in frontier_g:
                    for i in range(len(queue_s.queue)):
                        if front == queue_s.queue[i][2]:
                            break
                    for j in range(len(queue_g.queue)):
                        if front == queue_g.queue[j][2]:
                            break
                    try:
                        rval = queue_s.queue[i][0] + queue_g.queue[j][0]
                        communicated["node"].append(front)
                        communicated["cost_s"].append(queue_s.queue[i][0])
                        communicated["cost_g"].append(queue_g.queue[j][0])
                        communicated["min_cost"].append(rval)
                        if rval < min_cost:
                            min_cost = rval
                            min_front = front
                            min_cost_s = queue_s.queue[i][0]
                            min_cost_g = queue_g.queue[j][0]
                    except:
                        pass


        if len(queue_g.queue) > 0:
            (cost, _, node) = queue_g.pop()
            if node in communicated["node"] or cost >= min_cost_g:
                continue
            explored.append(node)
            if node in frontier_g:
                frontier_g.remove(node)

            queue_g, new_frontier, updated_frontier = bi_ucs_step(graph, node, path, explored, frontier_g, cost,
                                                                  queue_g, policy_s, policy_g, 'b')
            for front in updated_frontier:
                if front in communicated["node"]:
                    for i in range(len(queue_g.queue)):
                        if front == queue_g.queue[i][2]:
                            break
                    k = communicated["node"].index(front)
                    if queue_g.queue[i][0] < communicated["cost_g"][k]:
                        communicated["cost_g"][k] = queue_g.queue[i][0]
                        communicated["min_cost"][k] = queue_g.queue[i][0] + communicated["cost_s"][k]
                    if communicated["min_cost"][k] < min_cost:
                        min_cost = communicated["min_cost"][k]
                        min_front = front
                        min_cost_s = communicated["cost_s"][k]
                        min_cost_g = communicated["cost_g"][k]

            for front in new_frontier:
                if front in frontier_s:
                    for i in range(len(queue_s.queue)):
                        if front == queue_s.queue[i][2]:
                            break
                    for j in range(len(queue_g.queue)):
                        if front == queue_g.queue[j][2]:
                            break
                    try:
                        rval = queue_s.queue[i][0] + queue_g.queue[j][0]
                        communicated["node"].append(front)
                        communicated["cost_s"].append(queue_s.queue[i][0])
                        communicated["cost_g"].append(queue_g.queue[j][0])
                        communicated["min_cost"].append(rval)
                        if rval < min_cost:
                            min_cost = rval
                            min_front = front
                            min_cost_s = queue_s.queue[i][0]
                            min_cost_g = queue_g.queue[j][0]
                    except:
                        pass

    path_sel = findpathbi(policy_s, policy_g, start, goal, min_front)
    return path_sel

def findpathbi(policy_s, policy_g, start, goal, front):

    path_f = [front]
    child = front
    while child != start:
        for key, value in policy_s.items():
            if child in value:
                path_f.append(key)
                child = key
                break

    path_b = []
    child = front
    while child != goal:
        for key, value in policy_g.items():
            if child in value:
                path_b.append(key)
                child = key
                break

    path_f.reverse()
    path = path_f + path_b

    return path

def ap_ucs_step(graph, node, path, explored, frontier, cost, queue, policy_s, policy_g, mode, start, goal):
    new_frontier = []
    updated_frontier = []
    neighbors = graph[node]
    count = next(counter1)
    print(count)
    for child, val in sorted(neighbors.items()):
        u_c = val['weight']
        hvalt = euclidean_dist_heuristic(graph, child, goal)
        hvals = euclidean_dist_heuristic(graph, child, start)
        fval = cost + u_c + (hvalt-hvals)/2.
        if child in explored:
            continue
        elif child in frontier:
            for i in range(len(queue.queue)):
                if queue.queue[i][3] == child:
                    if fval < queue.queue[i][0]:
                        queue.remove(queue.queue[i])
                        queue.append((fval ,cost + u_c, child))
                        if mode == 'f':
                            removechild(policy_s, child)
                            policyappend(policy_s, node, child)
                        elif mode == 'b':
                            removechild(policy_g, child)
                            policyappend(policy_g, node, child)
                        updated_frontier.append(child)
        else:
            frontier.append(child)
            queue.append((fval, cost + u_c, child))
            new_frontier.append(child)
            if mode == 'f':
                policyappend(policy_s, node, child)
            elif mode == 'b':
                policyappend(policy_g, node, child)

    return queue, new_frontier, updated_frontier

counter1 = itertools.count()
def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    path = []
    if start == goal:
        return path
    frontier_s = [start]
    explored = []
    frontier_g = [goal]
    queue_s = PriorityQueue()
    queue_s.append((0, 0, start))
    queue_g = PriorityQueue()
    queue_g.append((0, 0, goal))
    min_cost = 1e+20
    min_cost_s = 1e+20
    min_cost_g = 1e+20
    policy_s = {}
    policy_g = {}
    communicated = {"node": [], "cost_s": [], "cost_g": [], "min_cost": []}
    while (1):
        if len(queue_s.queue) == 0 and len(queue_g.queue) == 0:
            break
        fcost = 0.
        if len(queue_s.queue) > 0:
            fcost = queue_s.queue[0][0]
        bcost = 0
        if len(queue_g.queue) > 0:
            bcost = queue_g.queue[0][0]
        if fcost >= min_cost_s and bcost >= min_cost_g or \
                len(queue_s.queue) == 0 and bcost >= min_cost_g or \
                len(queue_g.queue) == 0 and fcost >= min_cost_s:
            break

        if len(queue_s.queue) > 0:
            (fval, _, cost, node) = queue_s.pop()
            if node in communicated["node"] or fval >= min_cost:
                continue
            explored.append(node)
            if node in frontier_s:
                frontier_s.remove(node)

            queue_s, new_frontier, updated_frontier = ap_ucs_step(graph, node, path, explored, frontier_s, cost,
                                                                  queue_s, policy_s, policy_g, 'f', start, goal)
            for front in updated_frontier:
                if front in communicated["node"]:
                    for i in range(len(queue_s.queue)):
                        if front == queue_s.queue[i][3]:
                            break
                    k = communicated["node"].index(front)
                    if queue_s.queue[i][0] < communicated["cost_s"][k]:
                        communicated["cost_s"][k] = queue_s.queue[i][0]
                        communicated["min_cost"][k] = queue_s.queue[i][0] + communicated["cost_g"][k]
                        communicated["node"][k] = front
                    if communicated["min_cost"][k] < min_cost:
                        min_cost = communicated["min_cost"][k]
                        min_front = front
                        min_cost_s = communicated["cost_s"][k]
                        min_cost_g = communicated["cost_g"][k]

            for front in new_frontier:
                if front in frontier_g:
                    for i in range(len(queue_s.queue)):
                        if front == queue_s.queue[i][3]:
                            break
                    for j in range(len(queue_g.queue)):
                        if front == queue_g.queue[j][3]:
                            break
                    try:
                        rval = queue_s.queue[i][0] + queue_g.queue[j][0]
                        communicated["node"].append(front)
                        communicated["cost_s"].append(queue_s.queue[i][0])
                        communicated["cost_g"].append(queue_g.queue[j][0])
                        communicated["min_cost"].append(rval)
                        if rval < min_cost:
                            min_cost = rval
                            min_front = front
                            min_cost_s = queue_s.queue[i][0]
                            min_cost_g = queue_g.queue[j][0]
                    except:
                        pass

        if len(queue_g.queue) > 0:
            (fval, _, cost, node) = queue_g.pop()
            if node in communicated["node"] or fval >= min_cost_g:
                continue
            explored.append(node)
            if node in frontier_g:
                frontier_g.remove(node)

            queue_g, new_frontier, updated_frontier = ap_ucs_step(graph, node, path, explored, frontier_g, cost,
                                                                  queue_g, policy_s, policy_g, 'b', goal, start)
            for front in updated_frontier:
                if front in communicated["node"]:
                    for i in range(len(queue_g.queue)):
                        if front == queue_g.queue[i][3]:
                            break
                    k = communicated["node"].index(front)
                    if queue_g.queue[i][0] < communicated["cost_g"][k]:
                        communicated["cost_g"][k] = queue_g.queue[i][0]
                        communicated["min_cost"][k] = queue_g.queue[i][0] + communicated["cost_s"][k]
                    if communicated["min_cost"][k] < min_cost:
                        min_cost = communicated["min_cost"][k]
                        min_front = front
                        min_cost_s = communicated["cost_s"][k]
                        min_cost_g = communicated["cost_g"][k]

            for front in new_frontier:
                if front in frontier_s:
                    for i in range(len(queue_s.queue)):
                        if front == queue_s.queue[i][3]:
                            break
                    for j in range(len(queue_g.queue)):
                        if front == queue_g.queue[j][3]:
                            break
                    try:
                        rval = queue_s.queue[i][0] + queue_g.queue[j][0]
                        communicated["node"].append(front)
                        communicated["cost_s"].append(queue_s.queue[i][0])
                        communicated["cost_g"].append(queue_g.queue[j][0])
                        communicated["min_cost"].append(rval)
                        if rval < min_cost:
                            min_cost = rval
                            min_front = front
                            min_cost_s = queue_s.queue[i][0]
                            min_cost_g = queue_g.queue[j][0]
                    except:
                        pass

    path_sel = findpathbi(policy_s, policy_g, start, goal, min_front)
    return path_sel

def tri_ucs_step(graph, node, explored, frontier, frontier_cost, cost, queue, policy, ig):
    new_frontier = []
    updated_frontier = []
    neighbors = graph[node]
    for child, val in sorted(neighbors.items()):
        u_c = val['weight']
        if child in explored["explored"]:
            continue
        elif child in frontier:
            for i in range(len(queue.queue)):
                if queue.queue[i][2] == child:
                    if cost + u_c < queue.queue[i][0]:
                        queue.remove(queue.queue[i])
                        queue.append((cost + u_c, child))
                        removechild(policy, child)
                        policyappend(policy, node, child)
                        updated_frontier.append(child)
                        index = frontier.index(child)
                        frontier_cost[index] = cost + u_c
        else:
            frontier.append(child)
            frontier_cost.append(cost + u_c)
            queue.append((cost + u_c, child))
            new_frontier.append(child)
            policyappend(policy, node, child)

    return new_frontier, updated_frontier

def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function

    if goals[0] == goals[1] == goals[2]:
        return []
    nf = len(goals)
    # for i in range(nf):
    #     if goals[i] == goals[(i+1)%3]:
    #         return bidirectional_ucs(graph, goals[i], goals[i+1])

    frontier = []
    frontier_cost = []
    queue = []
    ind_cost = []
    policy = []
    costi = []
    for i in range(nf):
        frontier.append([goals[i]])
        frontier_cost.append([0.])
        queue.append(PriorityQueue())
        queue[i].append((0, goals[i]))
        ind_cost.append(1e+20)
        policy.append({})
        costi.append(0)

    explored ={"explored": [],"cost":[]}
    mincost01 = 1e+20
    mincost02 = 1e+20
    mincost12 = 1e+20
    communicated = {"node": [], "goal1": [], "goal2": [], "cost1": [], "cost2": [], "min_cost": []}
    while (1):
        if len(queue[0].queue) == 0 and len(queue[1].queue) and len(queue[2].queue) == 0:
            break
        for i in range(nf):
            try:
                costi[i] = queue[i].queue[0][0]
            except:
                costi[i] = 10000.
        if costi[0] >= ind_cost[0] and costi[1] >= ind_cost[1] and costi[2] >= ind_cost[2]:
            break

        for ig in range(nf):
            if len(queue[ig].queue) > 0:
                (cost, _, node) = queue[ig].pop()
                for i in range(nf):
                    try:
                        costi[i] = queue[i].queue[0][0]
                    except:
                        costi[i] = 100000.
                if costi[0] >= ind_cost[0] and costi[1] >= ind_cost[1] and costi[2] >= ind_cost[2]:
                    break
                # if node in communicated["node"] or cost >= ind_cost[ig]:
                #     continue
                if node in explored["explored"]:
                    ind = explored["explored"].index(node)
                    if cost >= explored["cost"][ind]:
                        continue
                n = frontier[ig].index(node)
                ip1 = (ig+1) % 3
                if node in frontier[ip1]:
                    m = frontier[ip1].index(node)
                    if frontier_cost[ig][n] > frontier_cost[ip1][m]:
                        continue
                ip1 = (ig - 1) % 3
                if node in frontier[ip1]:
                    m = frontier[ip1].index(node)
                    if frontier_cost[ig][n] > frontier_cost[ip1][m]:
                        continue

                explored["explored"].append(node)
                explored["cost"].append(cost)
                #if node in frontier[ig]:
                #    frontier[ig].remove(node)

                new_frontier, updated_frontier = tri_ucs_step(graph, node, explored, frontier[ig], frontier_cost[ig], cost,
                                                                      queue[ig], policy[ig], ig)

                for front in updated_frontier:
                    if front in communicated["node"]:
                        for i in range(len(queue[ig].queue)):
                            if front == queue[ig].queue[i][2]:
                                break
                        k = communicated["node"].index(front)
                        goal1 = communicated["goal1"][k]
                        goal2 = communicated["goal2"][k]
                        cost1 = communicated["cost1"][k]
                        cost2 = communicated["cost2"][k]
                        cost = communicated["min_cost"][k]
                        if ig == goal1 and queue[ig].queue[i][0] < cost1:
                            communicated["cost1"][k] = queue[ig].queue[i][0]
                            communicated["min_cost"][k] = queue[ig].queue[i][0] + communicated["cost2"][k]
                        elif ig == goal2 and queue[ig].queue[i][0] < cost2:
                            communicated["cost2"][k] = queue[ig].queue[i][0]
                            communicated["min_cost"][k] = queue[ig].queue[i][0] + communicated["cost1"][k]
                        if goal1 == 0 and goal2 == 1:
                            if communicated["min_cost"][k] < mincost01:
                                mincost01 = communicated["min_cost"][k]
                                front01 = front
                                ind_cost[goal1] = communicated["cost1"][k]
                                ind_cost[goal2] = communicated["cost2"][k]
                        if goal1 == 1 and goal2 == 2:
                            if communicated["min_cost"][k] < mincost12:
                                mincost12 = communicated["min_cost"][k]
                                front12 = front
                                ind_cost[goal1] = communicated["cost1"][k]
                                ind_cost[goal2] = communicated["cost2"][k]
                        if goal1 == 0 and goal2 == 1:
                            if communicated["min_cost"][k] < mincost02:
                                mincost02 = communicated["min_cost"][k]
                                front02 = front
                                ind_cost[goal1] = communicated["cost1"][k]
                                ind_cost[goal2] = communicated["cost2"][k]


                # Check if the front is also the front of another path.
                for front in new_frontier:
                    for inc in [-1,1]:
                        ip1 = (ig+inc) % nf
                        if front in frontier[ip1]:
                            for i in range(len(queue[ig].queue)):
                                if front == queue[ig].queue[i][2]:
                                    break
                            for j in range(len(queue[ip1].queue)):
                                if front == queue[ip1].queue[j][2]:
                                    break
                            try:
                                if ig < ip1:
                                    goal1 = ig
                                    goal2 = ip1
                                    i1 = i
                                    j1 = j
                                else:
                                    goal1 = ip1
                                    goal2 = ig
                                    i1 = j
                                    j1 = i
                                rval = queue[goal1].queue[i1][0] + queue[goal2].queue[j1][0]
                                communicated["node"].append(front)
                                communicated["goal1"].append(goal1)
                                communicated["goal2"].append(goal2)
                                communicated["cost1"].append(queue[goal1].queue[i1][0])
                                communicated["cost2"].append(queue[goal2].queue[j1][0])
                                communicated["min_cost"].append(rval)
                                if goal1 == 0 and goal2 == 1:
                                    if rval < mincost01:
                                        mincost01 = rval
                                        front01 = front
                                        ind_cost[goal1] = queue[goal1].queue[i1][0]
                                        ind_cost[goal2] = queue[goal2].queue[j1][0]
                                        # queue[ip1].remove(queue[ip1].queue[j])
                                    else:
                                        queue[ig].remove(queue[ig].queue[i])
                                elif goal1 == 0 and goal2 == 2:
                                    if rval < mincost02:
                                        mincost02 = rval
                                        front02 = front
                                        ind_cost[goal1] = queue[goal1].queue[i1][0]
                                        ind_cost[goal2] = queue[goal2].queue[j1][0]
                                        # queue[ip1].remove(queue[ip1].queue[j])
                                    else:
                                        queue[ig].remove(queue[ig].queue[i])
                                elif goal1 == 1 and goal2 == 2:
                                    if rval < mincost12:
                                        mincost12 = rval
                                        front12 = front
                                        ind_cost[goal1] = queue[goal1].queue[i1][0]
                                        ind_cost[goal2] = queue[goal2].queue[j1][0]
                                        # queue[ip1].remove(queue[ip1].queue[j])
                                    else:
                                        queue[ig].remove(queue[ig].queue[i])
                            except:
                                pass



    path1 = mincost01 + mincost02
    path2 = mincost01 + mincost12
    path3 = mincost02 + mincost12
    if path1 == min(path1,path2,path3):
        beg = 1
        f1 = front01
        mid = 0
        f2 = front02
        end = 2
    elif path2 == min(path1,path2,path3):
        beg = 0
        f1 = front01
        mid = 1
        f2 = front12
        end = 2
    elif path3 == min(path1,path2,path3):
        beg = 0
        f1 = front02
        mid = 2
        f2 = front12
        end = 1

    path_sel = findpathtri(policy, goals, beg, f1, mid, f2, end)
    return path_sel

def findpathtri(policy, goals, beg, f1, mid, f2, end):


    path1 = []
    start = goals[beg]
    child = f1
    while child != start:
        for key, value in policy[beg].items():
            if child in value:
                path1.append(key)
                child = key
                break
    path1.reverse()

    path2 = []
    child = f1
    path2.append(f1)
    while child != goals[mid]:
        for key, value in policy[mid].items():
            if child in value:
                path2.append(key)
                child = key
                break

    path3 = []
    child = f2
    while child != goals[mid]:
        for key, value in policy[mid].items():
            if child in value:
                path3.append(key)
                child = key
                break
    if f2 != goals[mid]:
        path3.remove(goals[mid])
    path3.reverse()


    path4 = []
    if f2 != goals[mid]:
        path4.append(f2)
    child = f2
    while child != goals[end]:
        for key, value in policy[end].items():
            if child in value:
                path4.append(key)
                child = key
                break

    path = path1 + path2 + path3 + path4

    return path

def euclidean_dist_heuristic2(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """
    a = graph.nodes[v]['pos']
    b = goal
    dist = (a[0] - b[0]) ** 2.0 + (a[1] - b[1]) ** 2.0
    dist = dist ** 0.5
    # TODO: finish this function!
    return dist

def tri_ucs_step_up(graph, node, explored, frontier, frontier_cost, cost, queue, policy, ig, goals):
    new_frontier = []
    updated_frontier = []
    neighbors = graph[node]

    for child, val in sorted(neighbors.items()):
        u_c = val['weight']
        hval1 = euclidean_dist_heuristic(graph, child, goals[(ig+1)%3])
        hval2 = euclidean_dist_heuristic(graph, child, goals[(ig-1)%3])
        if hval1 > hval2:
            # g = goals[(ig+1)%3]
            hvalt = hval1
        else:
            # g = goals[(ig-1)%3]
            hvalt = hval2
        # hvalt = euclidean_dist_heuristic(graph, child, g)
        hvals = euclidean_dist_heuristic(graph, child, goals[ig])
        fval = cost + u_c + (hvalt - hvals) / 2
        if child in explored["explored"]:
            continue
        elif child in frontier:
            for i in range(len(queue.queue)):
                if queue.queue[i][3] == child:
                    if cost + u_c < queue.queue[i][2]:
                        queue.remove(queue.queue[i])
                        queue.append((fval, cost + u_c, child))
                        removechild(policy, child)
                        policyappend(policy, node, child)
                        updated_frontier.append(child)
                        index = frontier.index(child)
                        frontier_cost[index] = cost + u_c
        else:
            frontier.append(child)
            frontier_cost.append(cost + u_c)
            queue.append((fval, cost + u_c, child))
            new_frontier.append(child)
            policyappend(policy, node, child)

    return new_frontier, updated_frontier

def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic, landmarks=None):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
        landmarks: Iterable containing landmarks pre-computed in compute_landmarks()
            Default: None

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    if goals[0] == goals[1] == goals[2]:
        return []
    nf = len(goals)
    # for i in range(nf):
    #     if goals[i] == goals[(i+1)%3]:
    #         return bidirectional_ucs(graph, goals[i], goals[i+1])

    frontier = []
    frontier_cost = []
    queue = []
    ind_cost = []
    policy = []
    costi = []
    for i in range(nf):
        frontier.append([goals[i]])
        frontier_cost.append([0.])
        queue.append(PriorityQueue())
        queue[i].append((0, 0, goals[i]))
        ind_cost.append(1e+20)
        policy.append({})
        costi.append(0)

    explored = {"explored": [], "cost": []}
    mincost01 = 1e+20
    mincost02 = 1e+20
    mincost12 = 1e+20
    communicated = {"node": [], "goal1": [], "goal2": [], "cost1": [], "cost2": [], "min_cost": []}
    while (1):
        if len(queue[0].queue) == 0 and len(queue[1].queue) and len(queue[2].queue) == 0:
            break
        for i in range(nf):
            try:
                costi[i] = queue[i].queue[0][2]
            except:
                costi[i] = 10000.
        if costi[0] >= ind_cost[0] and costi[1] >= ind_cost[1] and costi[2] >= ind_cost[2]:
            break

        for ig in range(nf):
            if len(queue[ig].queue) > 0:
                (fval, _, cost, node) = queue[ig].pop()
                for i in range(nf):
                    try:
                        costi[i] = queue[i].queue[0][2]
                    except:
                        costi[i] = 100000.
                if costi[0] >= ind_cost[0] and costi[1] >= ind_cost[1] and costi[2] >= ind_cost[2]:
                    break
                # if node in communicated["node"] or cost >= ind_cost[ig]:
                #     continue
                if node in explored["explored"]:
                    ind = explored["explored"].index(node)
                    if cost >= explored["cost"][ind]:
                        continue
                n = frontier[ig].index(node)
                ip1 = (ig + 1) % 3
                if node in frontier[ip1]:
                    m = frontier[ip1].index(node)
                    if frontier_cost[ig][n] > frontier_cost[ip1][m]:
                        continue
                ip1 = (ig - 1) % 3
                if node in frontier[ip1]:
                    m = frontier[ip1].index(node)
                    if frontier_cost[ig][n] > frontier_cost[ip1][m]:
                        continue

                explored["explored"].append(node)
                explored["cost"].append(cost)
                # if node in frontier[ig]:
                #    frontier[ig].remove(node)

                new_frontier, updated_frontier = tri_ucs_step_up(graph, node, explored, frontier[ig], frontier_cost[ig],
                                                              cost,
                                                              queue[ig], policy[ig], ig, goals)

                for front in updated_frontier:
                    if front in communicated["node"]:
                        for i in range(len(queue[ig].queue)):
                            if front == queue[ig].queue[i][3]:
                                break
                        k = communicated["node"].index(front)
                        goal1 = communicated["goal1"][k]
                        goal2 = communicated["goal2"][k]
                        cost1 = communicated["cost1"][k]
                        cost2 = communicated["cost2"][k]
                        cost = communicated["min_cost"][k]
                        if ig == goal1 and queue[ig].queue[i][2] < cost1:
                            communicated["cost1"][k] = queue[ig].queue[i][2]
                            communicated["min_cost"][k] = queue[ig].queue[i][2] + communicated["cost2"][k]
                        elif ig == goal2 and queue[ig].queue[i][2] < cost2:
                            communicated["cost2"][k] = queue[ig].queue[i][2]
                            communicated["min_cost"][k] = queue[ig].queue[i][2] + communicated["cost1"][k]
                        if goal1 == 0 and goal2 == 1:
                            if communicated["min_cost"][k] < mincost01:
                                mincost01 = communicated["min_cost"][k]
                                front01 = front
                                ind_cost[goal1] = communicated["cost1"][k]
                                ind_cost[goal2] = communicated["cost2"][k]
                        if goal1 == 1 and goal2 == 2:
                            if communicated["min_cost"][k] < mincost12:
                                mincost12 = communicated["min_cost"][k]
                                front12 = front
                                ind_cost[goal1] = communicated["cost1"][k]
                                ind_cost[goal2] = communicated["cost2"][k]
                        if goal1 == 0 and goal2 == 1:
                            if communicated["min_cost"][k] < mincost02:
                                mincost02 = communicated["min_cost"][k]
                                front02 = front
                                ind_cost[goal1] = communicated["cost1"][k]
                                ind_cost[goal2] = communicated["cost2"][k]

                # Check if the front is also the front of another path.
                for front in new_frontier:
                    for inc in [-1, 1]:
                        ip1 = (ig + inc) % nf
                        if front in frontier[ip1]:
                            for i in range(len(queue[ig].queue)):
                                if front == queue[ig].queue[i][3]:
                                    break
                            for j in range(len(queue[ip1].queue)):
                                if front == queue[ip1].queue[j][3]:
                                    break
                            try:
                                if ig < ip1:
                                    goal1 = ig
                                    goal2 = ip1
                                    i1 = i
                                    j1 = j
                                else:
                                    goal1 = ip1
                                    goal2 = ig
                                    i1 = j
                                    j1 = i
                                rval = queue[goal1].queue[i1][2] + queue[goal2].queue[j1][2]
                                communicated["node"].append(front)
                                communicated["goal1"].append(goal1)
                                communicated["goal2"].append(goal2)
                                communicated["cost1"].append(queue[goal1].queue[i1][2])
                                communicated["cost2"].append(queue[goal2].queue[j1][2])
                                communicated["min_cost"].append(rval)
                                if goal1 == 0 and goal2 == 1:
                                    if rval < mincost01:
                                        mincost01 = rval
                                        front01 = front
                                        ind_cost[goal1] = queue[goal1].queue[i1][2]
                                        ind_cost[goal2] = queue[goal2].queue[j1][2]
                                        # queue[ip1].remove(queue[ip1].queue[j])
                                    else:
                                        queue[ig].remove(queue[ig].queue[i])
                                elif goal1 == 0 and goal2 == 2:
                                    if rval < mincost02:
                                        mincost02 = rval
                                        front02 = front
                                        ind_cost[goal1] = queue[goal1].queue[i1][2]
                                        ind_cost[goal2] = queue[goal2].queue[j1][2]
                                        # queue[ip1].remove(queue[ip1].queue[j])
                                    else:
                                        queue[ig].remove(queue[ig].queue[i])
                                elif goal1 == 1 and goal2 == 2:
                                    if rval < mincost12:
                                        mincost12 = rval
                                        front12 = front
                                        ind_cost[goal1] = queue[goal1].queue[i1][2]
                                        ind_cost[goal2] = queue[goal2].queue[j1][2]
                                        # queue[ip1].remove(queue[ip1].queue[j])
                                    else:
                                        queue[ig].remove(queue[ig].queue[i])
                            except:
                                pass

    path1 = mincost01 + mincost02
    path2 = mincost01 + mincost12
    path3 = mincost02 + mincost12
    if path1 == min(path1, path2, path3):
        beg = 1
        f1 = front01
        mid = 0
        f2 = front02
        end = 2
    elif path2 == min(path1, path2, path3):
        beg = 0
        f1 = front01
        mid = 1
        f2 = front12
        end = 2
    elif path3 == min(path1, path2, path3):
        beg = 0
        f1 = front02
        mid = 2
        f2 = front12
        end = 1

    path_sel = findpathtri(policy, goals, beg, f1, mid, f2, end)
    return path_sel


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return 'Emre'


def compute_landmarks(graph):
    """
    Feel free to implement this method for computing landmarks. We will call
    tridirectional_upgraded() with the object returned from this function.

    Args:
        graph (ExplorableGraph): Undirected graph to search.

    Returns:
    List with not more than 4 computed landmarks. 
    """
    return None


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """
    pass


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to Gradescope, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None


def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    # Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    # Now we want to execute portions of the formula:
    constOutFront = 2 * 6371  # Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0] - vLatLong[0]) / 2)) ** 2  # First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0]) * math.cos(goalLatLong[0]) * (
            (math.sin((goalLatLong[1] - vLatLong[1]) / 2)) ** 2)  # Second term
    return constOutFront * math.asin(math.sqrt(term1InSqrt + term2InSqrt))  # Straight application of formula
