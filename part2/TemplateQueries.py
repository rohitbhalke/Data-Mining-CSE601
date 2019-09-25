from part2.AssociationRules import AssociationRuleGenerator as rule_gen

def readFile(path):
    data = []
    with open(path) as fp:
        line = fp.readline()
        while line:
            line = line.replace("\n", "")
            data.append(line)
            line = fp.readline()
    return data

def template_1_queries(rules):
    f = open("output.txt", "a")
    f.write("\n\nQueries for template 1")
    with open("queries1.txt") as fp:
        line = fp.readline()
        #print("File: ")
        while line:
            rule_list_result = []
            line = line.replace("\n", "")
            template_1_queries_find_rules(rule_list_result,line,rules)
            # print("Query: " + str(list[0]) + ", " + str(list[1]) + ", " + str(list[2]))
            f.write("\nQuery: " + line)
            f.write("\nNumber of rules:  " + str(len(rule_list_result)))
            print("Query: " + line)
            print("Number of rules:  " + str(len(rule_list_result)))
            for rule in rule_list_result:
                first = ",".join(rule[0])
                second = ",".join(rule[1])
                output = "{" + first + "->" + second + "}"
                f.write("\n" + output)
                print(output)
            line = fp.readline()
    f.close()

def template_1_queries_find_rules(rule_list_result, line, rules):
    list = line.split(" ")
    # print(list[0])
    # print(list[1])
    item_list = list[2][1:len(list[2]) - 1].split(",")
    # print(item_list)
    rule_list = []
    if list[1] == "ANY":
        if list[0] == "RULE":
            for r in rules:
                head = str(r[0])[2:len(str(r[0])) - 2]
                body = str(r[1])[2:len(str(r[1])) - 2]
                if any(m in r[0] for m in item_list) or any(m in r[1] for m in item_list):
                    rule_list_result.append(r)

        if list[0] == "HEAD":
            for r in rules:
                head = str(r[0])[2:len(str(r[0])) - 2]
                body = str(r[1])[2:len(str(r[1])) - 2]
                if any(m in r[0] for m in item_list):
                    rule_list_result.append(r)

        if list[0] == "BODY":
            for r in rules:
                head = str(r[0])[2:len(str(r[0])) - 2]
                body = str(r[1])[2:len(str(r[1])) - 2]
                if any(m in r[1] for m in item_list):
                    rule_list_result.append(r)

    if list[1] == "NONE":
        if list[0] == "RULE":
            for r in rules:
                head = str(r[0])[2:len(str(r[0])) - 2]
                body = str(r[1])[2:len(str(r[1])) - 2]
                if not (any(m in r[0] for m in item_list) or any(m in r[1] for m in item_list)):
                    rule_list_result.append(r)

        if list[0] == "HEAD":
            for r in rules:
                head = str(r[0])[2:len(str(r[0])) - 2]
                body = str(r[1])[2:len(str(r[1])) - 2]
                if not (any(m in r[0] for m in item_list)):
                    rule_list_result.append(r)

        if list[0] == "BODY":
            for r in rules:
                head = str(r[0])[2:len(str(r[0])) - 2]
                body = str(r[1])[2:len(str(r[1])) - 2]
                if not (any(m in r[1] for m in item_list)):
                    rule_list_result.append(r)

    if list[1].isnumeric():
        if list[0] == "RULE":
            for r in rules:
                head = str(r[0])[2:len(str(r[0])) - 2]
                body = str(r[1])[2:len(str(r[1])) - 2]
                if (len(set(r[0]).intersection(set(item_list))) >= int(list[1])) or (
                        len(set(r[1]).intersection(set(item_list))) >= int(list[1])):
                    rule_list_result.append(r)

        if list[0] == "HEAD":
            for r in rules:
                head = str(r[0])[2:len(str(r[0])) - 2]
                body = str(r[1])[2:len(str(r[1])) - 2]
                if len(set(r[0]).intersection(set(item_list))) >= int(list[1]):
                    rule_list_result.append(r)

        if list[0] == "BODY":
            for r in rules:
                head = str(r[0])[2:len(str(r[0])) - 2]
                body = str(r[1])[2:len(str(r[1])) - 2]
                if len(set(r[1]).intersection(set(item_list))) >= int(list[1]):
                    rule_list_result.append(r)
    return rule_list_result

def template_2_queries(rules):
    f = open("output.txt", "a")
    f.write("\n\nQueries for template 2")
    file_path = "queries2.txt"
    dataset = readFile(file_path)
    for item in dataset:
        result = []
        result = template_2_queries_find_rules(result,item,rules)
        f.write("\nQuery: " + item)
        f.write("\nNumber of rules: " + str(len(result)))
        print("Query: " + item)
        print("Number of rules: " + str(len(result)))
        for rule in result:
            first = ",".join(rule[0])
            second = ",".join(rule[1])
            output = "{" + first + "->" + second + "}"
            print(output)
            f.write("\n" + output)
    f.close()

def template_2_queries_find_rules(result, item, rules):
    query_param = item.split(" ")
    key = query_param[0]
    req_count = query_param[1]
    for rule in rules:
        head_count = str(len(rule[0]))
        tail_count = str(len(rule[1]))
        rule_count = head_count + tail_count
        if (key == "RULE") and (rule_count == req_count):
            result.append(rule)
        elif (key == "HEAD") and (head_count == req_count):
            result.append(rule)
        elif (key == "BODY") and (tail_count == req_count):
            result.append(rule)
        else:
            pass
    return result

def template_3_queries(rules):
    f = open("output.txt", "a")
    f.write("\n\nQueries for template 3")
    file_path = "queries3.txt"
    dataset = readFile(file_path)
    for item in dataset:
        result = []
        result_1 = []
        result_2 = []
        query_param = item.split(" ")
        query = query_param[0]
        q1 = query[0:1]
        q2 = query[1:len(query)-1]
        q3 = query[len(query)-1:len(query)]
        is_1 = False
        if q1 == str(1):
            is_1 = True
            query_list = [query_param[1],query_param[2],query_param[3]]
            query_str = " ".join(query_list)
            template_1_queries_find_rules(result_1,query_str,rules)
        else:
            query_list = [query_param[1], query_param[2]]
            query_str = " ".join(query_list)
            template_2_queries_find_rules(result_1,query_str,rules)
        if q3 == str(1):
            if is_1:
                query_list = [query_param[4], query_param[5], query_param[6]]
            else:
                query_list = [query_param[3], query_param[4], query_param[5]]
            query_str = " ".join(query_list)
            template_1_queries_find_rules(result_2,query_str,rules)
        else:
            if is_1:
                query_list = [query_param[4], query_param[5]]
            else:
                query_list = [query_param[3], query_param[4]]
            query_str = " ".join(query_list)
            template_2_queries_find_rules(result_2,query_str,rules)

        if q2 == "or":
            result = result_1
            for rule in result_2:
                if rule in result:
                    continue
                else:
                    result.append(rule)
        else:
            for rule in result_1:
                if rule in result_2:
                    result.append(rule)

        f.write("\nQuery: " + item)

        print("Query: " + item)
        f.write("\nNumber of rules: " + str(len(result)))
        for rule in result:
            first = ",".join(rule[0])
            second = ",".join(rule[1])
            output = "{" + first + "->" + second + "}"
            print(output)
            f.write("\n" + output)
    f.close()


if __name__ == '__main__':
    filePath = "associationruletestdata.txt"
    association_rules = rule_gen.main(filePath,60)
    print("----Template 1----")
    template_1_queries(association_rules)
    print("----Template 2----")
    template_2_queries(association_rules)
    print("----Template 3----")
    template_3_queries(association_rules)
