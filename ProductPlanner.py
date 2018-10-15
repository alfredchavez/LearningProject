from random import randint, random
from copy import deepcopy


class ProductPlanner(object):
    def __init__(self, configuration):
        self.alpha = configuration["learning_rate"]
        self.gamma = configuration["discount_rate"]
        self.budget = configuration["budget"]
        self.products = {}
        self.products_idx = {}
        self.idx = 0
        self.mc_it = configuration["monte_carlo_iterations"]
        self.q_it = configuration["q_learning_iterations"]
        self.rmatrix = {}

    def add_product(self, product_name, product_type, product_cost):
        if product_type not in self.products:
            self.products[product_type] = []
            self.products_idx[self.idx] = product_type
            self.idx += 1
        self.products[product_type].append((product_name, product_cost))

    def initialize_matrix(self):
        for it in range(0, self.idx - 1):
            self.rmatrix[(it, it + 1)] = []
            for j in self.products[self.products_idx[it]]:
                rlist = []
                for k in self.products[self.products_idx[it + 1]]:
                    rlist.append(0)
                self.rmatrix[(it, it + 1)].append(rlist)

    def monte_carlo(self):
        for iteration in range(0, self.mc_it):
            c_state = -1
            q = []
            for it in range(0, self.idx - 1):
                if c_state == -1:
                    i_state = randint(0, len(self.products[self.products_idx[it]]) - 1)
                else:
                    i_state = deepcopy(c_state)
                # n_state = randint(0, len(self.products[self.products_idx[it + 1]]) - 1)
                if self.egreedy(0.9 - (0.9 - 0.1)/self.mc_it):
                    ind = 0
                    maxv = self.rmatrix[(it, it + 1)][i_state][0]
                    for idx, i in enumerate(self.rmatrix[(it, it + 1)][i_state]):
                        if maxv < i:
                            ind = deepcopy(idx)
                            maxv = deepcopy(ind)
                    n_state = ind
                else:
                    n_state = randint(0, len(self.products[self.products_idx[it + 1]]) - 1)
                q.append((it, it + 1, i_state, n_state))
                c_state = deepcopy(n_state)
            total_cost = self.products[self.products_idx[q[0][0]]][q[0][2]][1]
            for record in q:
                total_cost += self.products[self.products_idx[record[1]]][record[3]][1]
            if total_cost > self.budget:
                value = self.budget - total_cost
            else:
                value = self.budget - total_cost
            for record in q:
                self.rmatrix[(record[0], record[1])][record[2]][record[3]] += self.alpha * (
                        value - self.rmatrix[(record[0], record[1])][record[2]][record[3]])

    def egreedy(self, epsilon):
        num = random()
        if num < epsilon:
            return False
        else:
            return True

    def learn(self):
        qmatrix = deepcopy(self.rmatrix)
        for i in qmatrix:
            for r in range(0, len(qmatrix[i])):
                for s in range(0, len(qmatrix[i][r])):
                    qmatrix[i][r][s] = 0
        for iteration in range(0, self.q_it):
            c_state = -1
            for it in range(0, self.idx - 1):
                if c_state == -1:
                    i_state = randint(0, len(self.products[self.products_idx[it]]) - 1)
                else:
                    i_state = deepcopy(c_state)
                if self.egreedy(0.5 - (0.5 - 0.1)/self.q_it):
                    ind = 0
                    maxv = qmatrix[(it, it + 1)][i_state][0]
                    for idx, i in enumerate(qmatrix[(it, it + 1)][i_state]):
                        if maxv < i:
                            ind = deepcopy(idx)
                            maxv = deepcopy(ind)
                    n_state = ind
                else:
                    n_state = randint(0, len(self.products[self.products_idx[it + 1]]) - 1)
                maxim = -1 * self.budget - 100
                if it + 1 == self.idx - 1:
                    maxim = 0
                else:
                    for idx, el in enumerate(qmatrix[(it + 1, it + 2)]):
                        if idx == n_state:
                            for i in el:
                                maxim = max(maxim, i)
                            break
                q_change = self.rmatrix[(it, it + 1)][i_state][n_state] + self.gamma * maxim
                qmatrix[(it, it + 1)][i_state][n_state] += self.alpha * (
                            q_change - qmatrix[(it, it + 1)][i_state][n_state])

        return qmatrix

    def get_answer(self):
        initial_state = 0
        ans = []
        total_price = 0
        for iterator in range(1, self.idx):
            max = self.rmatrix[(iterator -1, iterator)][initial_state][0]
            index = 0
            for idx, val in enumerate(self.rmatrix[(iterator - 1, iterator)][initial_state]):
                if val > max:
                    max = val
                    index = idx
            ans.append((self.products_idx[iterator], self.products[self.products_idx[iterator]][index]))
            total_price += self.products[self.products_idx[iterator]][index][1]
            initial_state = index
        return total_price, ans


