
def prune_X(X, transitions, n_states):
    print("pruning X")
    X_result = set(X)
    # n_states = len(H.U)
    found = True
    while found:
        found = False
        for i in range(0, min(X_PRUNE_MAX, len(X_result))):
            i = random.randint(0, len(X_result)-1)
            X_candidate = list(X_result)
            X_candidate.pop(i)
            X_candidate = set(X_candidate)
            transitions_new, _ = consistent_hyp(X_candidate, n_states, report=False)
            # equivalent = True
            # for (labels, rewards) in X:
            #     H_output = rm_run(labels, H)
            #     H_new_output = rm_run(labels, H_new)
            #     if H_output != H_new_output:
            #         equivalent = False
            if isomorphic(transitions, transitions_new, n_states):
                found = True
                X_result = X_candidate
                print(f"removed counterexample ({len(X_result)}/{len(X)})")
                # exit()
                break
        if len(X_result) <= X_PRUNE_MIN_SIZE:
            break

    # X_result = set()
    # first = True
    # for (labels, rewards) in X:
    #     labels_set = set(labels)
    #     if labels_set != set(('', 'b')) or first:
    #         X_result.add((labels, rewards))
    #         first = False

    print(f"new size is {len(X_result)}")

    return X_result