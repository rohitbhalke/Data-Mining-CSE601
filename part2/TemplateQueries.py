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

def template_2_queries(rules):
    file_path = "queries2.txt"
    dataset = readFile(file_path)
    for item in dataset:
        result = []
        query_param = item.split(",")
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
        print("Result for query - " + item)
        print(len(result))
        for rule in result:
            first = ",".join(rule[0])
            second = ",".join(rule[1])
            output = first + "->" + second
            print(output)

if __name__ == '__main__':
    filePath = "associationruletestdata.txt"
    association_rules = rule_gen.main(filePath,50)
    template_2_queries(association_rules)