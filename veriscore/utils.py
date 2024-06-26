# get K median and K max
from collections import defaultdict

def get_stats(domain_model_triplet_dict):
    domain_K_dict = defaultdict(lambda: defaultdict(int))
    for domain, model_triplet_dict in domain_model_triplet_dict.items():
        claim_num_lst = []
        for model_name, triplet_lst in model_triplet_dict.items():
            for triplet in triplet_lst:
                claim_num_lst.append(triplet[1])

        claim_num_lst.sort()
        K_median = claim_num_lst[len(claim_num_lst)//2]
        K_max = claim_num_lst[-1]
        domain_K_dict[domain]["K_median"] = K_median
        domain_K_dict[domain]["K_max"] = K_max
        # print(f"{domain} - {K_median}: {K_max}")

    return domain_K_dict

def get_avg_numbers(domain_model_triplet_dict, domain_K_dict):
    for domain, model_triplet_dict in domain_model_triplet_dict.items():
        K_median = domain_K_dict[domain]["K_median"]
        K_max = domain_K_dict[domain]["K_max"]

        table_content = []
        F1_at_median_lst = []
        for model_name in model_triplet_dict.keys():
            triplet_lst = domain_model_triplet_dict[domain][model_name]

            sent_len_lst = [x[2] for x in triplet_lst]
            sup_lst = [x[0] for x in triplet_lst]
            uns_lst = [x[1] - x[0] for x in triplet_lst]
            prec_lst = [x[0] / x[1] for x in triplet_lst]
            rec_med_lst = [min(x[0] / K_median, 1) for x in triplet_lst]
            rec_max_lst = [min(x[0] / K_max, 1) for x in triplet_lst]

            # get f1@K median and f1@K max
            f1_med_lst = [2 * prec * rec_med / (prec + rec_med) if rec_med > 0 else 0 for prec, rec_med in zip(prec_lst, rec_med_lst)]
            f1_max_lst = [2 * prec * rec_max / (prec + rec_max) if rec_max > 0 else 0 for prec, rec_max in zip(prec_lst, rec_max_lst)]

            # get ave. numbers
            ave_sent = sum(sent_len_lst) / len(sent_len_lst)
            S = sum(sup_lst) / len(sup_lst)
            U = sum(uns_lst) / len(uns_lst)
            P = sum(prec_lst) / len(prec_lst)
            Rec_med = sum(rec_med_lst) / len(rec_med_lst)
            Rec_max = sum(rec_max_lst) / len(rec_max_lst)
            F1_med = sum(f1_med_lst) / len(f1_med_lst)
            F1_max = sum(f1_max_lst) / len(f1_max_lst)

            table_row = [model_name, domain, round(ave_sent, 3), round(S, 3), round(U, 3), round(P, 3), round(Rec_med, 3), round(Rec_max, 3), round(F1_med, 3), round(F1_max, 3)]
            table_content.append(table_row)

            F1_at_median_lst.append(100*round(F1_med, 3))

            print(f"[{domain}-{model_name}] \nF1@k median: {F1_med:.3f}, F1@k max: {F1_max:.3f}")

def get_veriscore(domain_model_triplet_dict):
     domain_K_dict= get_stats(domain_model_triplet_dict)
     get_avg_numbers(domain_model_triplet_dict, domain_K_dict)