# genetic algorithm search for continuous function optimization
from numpy.random import randint
from numpy.random import rand


# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
    decoded = list()
    largest = 2 ** n_bits
    for i in range(len(bounds)):
        # extract the substring
        start, end = i * n_bits, (i * n_bits) + n_bits
        substring = bitstring[start:end]
        # convert bitstring to a string of chars
        chars = "".join([str(s) for s in substring])
        # convert string to integer
        integer = int(chars, 2)
        # scale integer to desired range
        ratio = integer / largest
        value = bounds[i][0] + ratio * (bounds[i][1] - bounds[i][0])
        # store
        decoded.append(value)
    return decoded


# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k - 1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1) - 2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


# mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]


# genetic algorithm
def genetic_algorithm(
    objective=None,
    target=None,
    bounds=None,
    break_accuracy=0.005,
    digits=5,
    n_bits=16,
    n_iter=100,
    n_pop=100,
    r_cross=0.9,
    r_mut=None,
):
    """genetic algorithm will compute on the objectiv loss function
    and given bounds for the features in the loss function a suggestion
    for new values for the model or loss function

    Args:
        objective ([function]): a loss function
        target ([number]): target value to optimize for
        bounds ([list]): a list for lower and upper limits
        break_accuracy (float): Min Difference to break,
        Defaults to 0.005.
        digits (int): number of digits for solution
        displayed Defaults to 5.
        n_bits (int): number of bits for a number. Defaults to 16.
        n_iter (int): number for iterations. Defaults to 100.
        n_pop (int): number of solutions test per iteration.
        Defaults to 100.
        r_cross (float): value for intercrossing. Defaults to 0.9.
        r_mut ([type]): value for mutations.
        Defaults to None: r_mut = 1.0 / (float(n_bits) * len(bounds))

    Returns:
        [type]: [description]
    """
    if r_mut is None and bounds is not None:
        r_mut = 1.0 / (float(n_bits) * len(bounds))
    else:
        r_mut = 0.5
    # initial population of random bitstring
    pop = [randint(0, 2, n_bits * len(bounds)).tolist() for _ in range(n_pop)]
    # keep track of best solution
    best = 0
    best_eval = objective(target=target, X=decode(bounds, n_bits, pop[0]))
    # enumerate generations
    for gen in range(n_iter):
        if best_eval <= break_accuracy:
            break
        else:
            # decode population
            decoded = [decode(bounds, n_bits, p) for p in pop]
            # evaluate all candidates in the population
            scores = [objective(target=target, X=d) for d in decoded]
            # check for new best solution
            for i in range(n_pop):
                if scores[i] < best_eval:
                    best, best_eval = pop[i], scores[i]
                    # print(">%d, new best f(%s) =
                    # %f" % (gen, decoded[i], scores[i]))
                    rs = round(scores[i][0], digits)
                    print(f">{gen}, new best {decoded[i]} = {rs}")
            # select parents
            selected = [selection(pop, scores) for _ in range(n_pop)]
            # create the next generation
            children = list()
            for i in range(0, n_pop, 2):
                # get selected parents in pairs
                p1, p2 = selected[i], selected[i + 1]
                # crossover and mutation
                for c in crossover(p1, p2, r_cross):
                    # mutation
                    mutation(c, r_mut)
                    # store for next generation
                    children.append(c)
            # replace population
            pop = children
    print("Done!")
    # return [best, best_eval]
    decoded = decode(bounds, n_bits, best)
    rounded_values = [round(element, digits) for element in decoded]
    return rounded_values






def create_bounds_list(names_order, bounds_dict):
    return [bounds_dict[element] for element in names_order]





# bounds_dict = {
#     "Xf": [13.45, 18.4],
#     "SA": [52, 79.7],
#     "M_per": [0, 3.6],
# }  # out of mlflow
# create_df_testing_cnames = ["M_per", "Xf", "SA"]


# bounds = create_bounds_list(create_df_testing_cnames, bounds_dict)

# target = 196


# new_setpoints = genetic_algorithm(
#     objective=loss_MFI_function,
#     target=target,
#     bounds=bounds,
#     break_accuracy=0.005,
#     n_bits=16,
#     n_iter=100,
#     n_pop=100,
#     r_cross=0.9,
#     r_mut=None,
# )
# new_setpoints


# MFI_model.predict(create_df_testing(M_per=new_setpoints[0], Xf=new_setpoints[1], SA=new_setpoints[2]))[0]


# ####

# # Mutli parameter

# # objective function
# def loss_MFI_function(target, X):

#     M_per = X[0]
#     Xf = X[1]
#     SA = X[2]
#     idata = create_df_testing(M_per, Xf, SA)
#     modeloutput_MFI = MFI_model.predict(idata)
#     modeloutput_CI = CI_model.predict(idata)

#     diff = abs(target[0] - modeloutput_MFI)
#     diff2 = abs(target[1] - modeloutput_CI)
#     return 2*diff+diff2


# bounds_dict = {
#     "Xf": [13.45, 18.4],
#     "SA": [52, 79.7],
#     "M_per": [0, 3.6],
# }  # out of mlflow
# create_df_testing_cnames = ["M_per", "Xf", "SA"]


# bounds = create_bounds_list(create_df_testing_cnames, bounds_dict)
# bounds

# target = [196, 90]


# new_setpoints = genetic_algorithm(
#     objective=loss_MFI_function,
#     target=target,
#     bounds=bounds,
#     break_accuracy=0.005,
#     n_bits=16,
#     n_iter=100,
#     n_pop=100,
#     r_cross=0.9,
#     r_mut=None,
# )
# new_setpoints # 1.614, 17.87, 65.15  #0.22148, 14.188, 70,318



# MFI_model.predict(create_df_testing(M_per=new_setpoints[0], Xf=new_setpoints[1], SA=new_setpoints[2]))[0]
# #MFI = 196.8

# CI_model.predict(create_df_testing(M_per=new_setpoints[0], Xf=new_setpoints[1], SA=new_setpoints[2]))[0]
# #CI = 98.4

# new_setpoints= [1.614, 17.87, 65.15]
# loss_MFI_function(target=[196, 90], X =new_setpoints)

# MFI_model.predict(create_df_testing(M_per=new_setpoints[0], Xf=new_setpoints[1], SA=new_setpoints[2]))[0]
# CI_model.predict(create_df_testing(M_per=new_setpoints[0], Xf=new_setpoints[1], SA=new_setpoints[2]))[0]





