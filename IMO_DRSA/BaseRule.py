import types

class BaseRule():
    """
    Example usage:
    rules = BaseRule()
    rules.create_function('greet', ['name'], "print(f'Hello, {name}!')")

    rules.greet("world")  # Output: Hello, world!
    print(obj.list_rules())  # Output: ['greet']
    """

    def __init__(self):
        self._rules = {}


    def check_all_rules(self, x) -> bool:

        for rule in self._rules:
            rule(x) # TODO: this is poopy

        return False

    def create_rule(self, name, args, body):
        """
        Create a function dynamically and bind it as a rule.



        :param name:
        :param args:
        :param body:
        :return:
        """
        args_str = ', '.join(args)
        func_code = f"def {name}(self, {args_str}):\n"
        func_code += '\n'.join(f"    {line}" for line in body.split('\n'))

        local_ns = {}
        exec(func_code, globals(), local_ns)

        func = local_ns[name]
        bound_rule = types.MethodType(func, self)
        setattr(self, name, bound_rule)
        self._rules[name] = bound_rule


    def list_rules(self):
        return list(self._rules.keys())