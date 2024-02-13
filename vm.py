"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import dis
import operator
import types
import typing as tp

from arg_binding import bind_args


BINARY_OPERATORS: list[tp.Callable[[tp.Any, tp.Any], tp.Any]] = [
    operator.add,
    operator.and_,
    operator.floordiv,
    operator.lshift,
    operator.matmul,
    operator.mul,  # 5
    operator.mod,
    operator.or_,
    operator.pow,
    operator.rshift,
    operator.sub,  # 10
    operator.truediv,
    operator.xor,
    operator.iadd,  # 13
    operator.iand,
    operator.ifloordiv,  # 15
    operator.ilshift,
    operator.imatmul,
    operator.imul,
    operator.imod,
    operator.ior,  # 20
    operator.ipow,
    operator.irshift,
    operator.isub,
    operator.itruediv,
    operator.ixor  # 25
]

COMPARE_OPERATORS: dict[str, tp.Callable[[tp.Any, tp.Any], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge  # 5
}


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.11/Include/frameobject.h

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """

    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: dict[str, tp.Any],
                 frame_globals: dict[str, tp.Any],
                 frame_locals: dict[str, tp.Any]) -> None:
        self.code = frame_code

        self.instructions: list[dis.Instruction] = []
        self.jump_offset_to_pos: tp.Mapping[int, int] = {}
        self.instruction_pointer: int = 0
        self._setup_instructions()

        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.kw_names: tp.Tuple = tuple()
        self.return_value = None
        self.has_return_value = False
        self.yield_value = None
        self.has_yield_value = False

    def __iter__(self) -> tp.Any:
        return self

    def __next__(self) -> tp.Any:
        while not self.has_yield_value or self.has_return_value:
            self.run_one_instruction()
        if self.has_return_value:
            raise StopIteration
        self.has_yield_value = False
        return self.yield_value

    def _setup_instructions(self) -> None:
        self.instructions = list(dis.get_instructions(self.code))
        self.jump_offset_to_pos = {
            instruction.offset: pos
            for pos, instruction in enumerate(self.instructions)
            if instruction.is_jump_target
        }
        self.instruction_pointer = 0

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        return self.data_stack.pop()

    def get_nth(self, n: int) -> tp.Any:
        return self.data_stack[~n]

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def pop_n(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def run(self) -> tp.Any:
        while not self.has_return_value:
            self.run_one_instruction()
        return self.return_value

    def run_one_instruction(self) -> None:
        instruction = self.instructions[self.instruction_pointer]
        previous_instruction_pointer = self.instruction_pointer
        getattr(self, instruction.opname.lower() + "_op")(instruction.argval)
        if previous_instruction_pointer == self.instruction_pointer:
            # if no jump happened in operator, increase position
            self.instruction_pointer += 1

    # instruction.argval for jump operators contains absolute offset
    # since then all jump operators look identical
    # instruction.arg could be used, but it doesn't contain other fancy values
    # f.e. variable names represented as strings, not as integer values

    def _jump(self, delta: int) -> None:
        self.instruction_pointer = self.jump_offset_to_pos[delta]

    jump_absolute_op = _jump
    jump_forward_op = _jump
    jump_backward_op = _jump

    def _pop_jump_if_true(self, delta: int) -> None:
        if self.pop():
            self._jump(delta)

    pop_jump_forward_if_true_op = _pop_jump_if_true
    pop_jump_backward_if_true_op = _pop_jump_if_true

    def _pop_jump_if_false(self, delta: int) -> None:
        if not self.pop():
            self._jump(delta)

    pop_jump_forward_if_false_op = _pop_jump_if_false
    pop_jump_backward_if_false_op = _pop_jump_if_false

    def _jump_if_none(self, delta: int) -> None:
        if self.pop() is None:
            self._jump(delta)

    pop_jump_forward_if_none_op = _jump_if_none
    pop_jump_backward_if_none_op = _jump_if_none

    def _jump_if_not_none(self, delta: int) -> None:
        if self.pop() is not None:
            self._jump(delta)

    pop_jump_forward_if_not_none_op = _jump_if_not_none
    pop_jump_backward_if_not_none_op = _jump_if_not_none

    def jump_if_true_or_pop_op(self, delta: int) -> None:
        if self.top():
            self._jump(delta)
        else:
            self.pop()

    def jump_if_false_or_pop_op(self, delta: int) -> None:
        if not self.top():
            self._jump(delta)
        else:
            self.pop()

    def for_iter_op(self, delta: int) -> None:
        try:
            next_val = next(self.top())
            self.push(next_val)
        except StopIteration:
            self.pop()
            self._jump(delta)

    def resume_op(self, arg: int) -> tp.Any:
        pass

    def push_null_op(self, arg: int) -> tp.Any:
        self.push(None)

    def precall_op(self, arg: int) -> tp.Any:
        pass

    def call_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-CALL
        """
        arg1, arg2, *args = self.pop_n(arg + 2)
        kw_count = len(self.kw_names)
        if kw_count:
            kwargs = dict(zip(self.kw_names, args[-kw_count:]))
            self.kw_names = tuple()
            args = args[:-kw_count]
        else:
            kwargs = {}
        if arg1 is None:
            try:
                self.push(arg2(*args, **kwargs))
            except TypeError:
                self.push(getattr(arg2, "__init__")(*args, **kwargs))
        else:
            self.push(arg1(arg2, *args, **kwargs))

    def call_function_ex_op(self, argval: int) -> None:
        arg = self.instructions[self.instruction_pointer].arg
        if arg is not None and arg & 1:
            kwargs = self.pop()
        else:
            kwargs = {}
        args = self.pop()
        callable_ = self.pop()
        self.pop()
        self.push(callable_(*args, **kwargs))

    def before_with_op(self) -> None:
        self.push(self.builtins["__exit__"])
        self.push(self.builtins["__enter__"]())

    def kw_names_op(self, argval: tp.Any) -> None:
        arg = self.instructions[self.instruction_pointer].arg
        if arg is None:
            return None
        self.kw_names = self.code.co_consts[arg]

    def copy_op(self, i: int) -> None:
        self.push(self.data_stack[-i])

    def swap_op(self, i: int) -> None:
        self.data_stack[-1], self.data_stack[-i] = self.data_stack[-i], self.data_stack[-1]

    def load_assertion_error_op(self, arg: tp.Any) -> None:
        self.push(AssertionError)

    def raise_varargs_op(self, flag: int) -> None:
        pass

    def load_method_op(self, name: str) -> None:
        class_ = self.pop()
        self.push(getattr(type(class_), name))
        self.push(class_)

    def load_name_op(self, arg: str) -> None:
        """
        Partial realization

        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_NAME
        """
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError(arg + " not found")

    def load_fast_op(self, arg: str) -> None:
        if arg not in self.locals:
            raise UnboundLocalError(arg)
        self.push(self.locals[arg])

    def store_fast_op(self, arg: str) -> None:
        self.locals[arg] = self.pop()

    def delete_fast_op(self, arg: str) -> None:
        if arg not in self.locals:
            pass
        del self.locals[arg]

    def load_global_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_GLOBAL
        """
        if (real_arg := self.instructions[self.instruction_pointer].arg) and real_arg & 1:
            self.push(None)
        if arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError(arg + " not found")

    def load_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_CONST
        """
        self.push(arg)

    def load_attr_op(self, name: str) -> None:
        self.push(getattr(self.pop(), name))

    def store_attr_op(self, name: str) -> None:
        setattr(self.pop(), name, self.pop())

    def delete_attr_op(self, name: str) -> None:
        delattr(self.pop(), name)

    def return_value_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-RETURN_VALUE
        """
        self.has_return_value = True
        self.return_value = self.pop()

    def yield_value_op(self, arg: tp.Any) -> None:
        self.has_yield_value = True
        self.yield_value = self.pop()

    def pop_top_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-POP_TOP
        """
        self.pop()

    def import_name_op(self, arg: str) -> None:
        tos = self.pop()
        tos1 = self.pop()
        self.push(__import__(arg, fromlist=tos, level=tos1))

    def import_from_op(self, arg: str) -> None:
        self.push(getattr(self.top(), arg))

    def import_star_op(self, arg: tp.Any) -> None:
        lib = self.pop()
        for attr in dir(lib):
            if not attr.startswith("_"):
                self.locals[attr] = getattr(lib, attr)

    def make_function_op(self, flags: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-MAKE_FUNCTION
        """
        code = self.pop()  # the code associated with the function (at TOS1)

        if flags & 2:
            kw_defaults = self.pop()
        else:
            kw_defaults = {}

        if flags & 1:
            default_values = self.pop()
        else:
            default_values = ()

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            parsed_args: dict[str, tp.Any] = bind_args(code, default_values, kw_defaults, *args, **kwargs)
            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            frame = Frame(code, self.builtins, self.globals, f_locals)  # Run code in prepared environment
            return frame.run()

        self.push(f)

    def store_name_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-STORE_NAME
        """
        const = self.pop()
        self.locals[arg] = const

    def delete_name_op(self, arg: str) -> None:
        del self.locals[arg]

    def store_global_op(self, arg: str) -> None:
        self.globals[arg] = self.pop()

    def delete_global_op(self, arg: str) -> None:
        del self.globals[arg]

    def unary_positive_op(self, arg: tp.Any) -> None:
        self.push(+self.pop())

    def unary_negative_op(self, arg: tp.Any) -> None:
        self.push(-self.pop())

    def unary_not_op(self, arg: tp.Any) -> None:
        self.push(not self.pop())

    def unary_invert_op(self, arg: tp.Any) -> None:
        self.push(~self.pop())

    def get_iter_op(self, arg: tp.Any) -> None:
        self.push(iter(self.pop()))

    def binary_subscr_op(self, arg: tp.Any) -> None:
        tos = self.pop()
        tos1 = self.pop()
        self.push(tos1[tos])

    def store_subscr_op(self, arg: tp.Any) -> None:
        tos = self.pop()
        tos1 = self.pop()
        tos2 = self.pop()
        tos1[tos] = tos2

    def delete_subscr_op(self, arg: tp.Any) -> None:
        tos = self.pop()
        tos1 = self.pop()
        del tos1[tos]

    def build_tuple_op(self, count: int) -> None:
        self.push(tuple(self.pop_n(count)))

    def build_list_op(self, count: int) -> None:
        self.push(list(self.pop_n(count)))

    def list_extend_op(self, i: int) -> None:
        assert i < 2, "wtf??"
        iterable = self.pop()
        self.top().extend(iterable)

    def build_map_op(self, count: int) -> None:
        new_dict = dict()
        for _ in range(count):
            key, val = self.pop_n(2)
            new_dict[key] = val
        self.push(new_dict)

    def build_const_key_map_op(self, count: int) -> None:
        keys = self.pop()
        self.push(dict(zip(keys, self.pop_n(count))))

    def build_set_op(self, count: int) -> None:
        self.push(set(self.pop_n(count)))

    def set_add_op(self, i: int) -> None:
        val = self.pop()
        set.add(self.data_stack[-i], val)

    def set_update_op(self, i: int) -> None:
        val = self.pop()
        set.update(self.data_stack[-i], val)

    def dict_update_op(self, i: int) -> None:
        val = self.pop()
        dict.update(self.data_stack[-i], val)

    def dict_merge_op(self, i: int) -> None:
        first = self.top()
        second = self.data_stack[-i]
        if first.keys().isdisjoint(second.keys()):
            # TODO: Pick Error
            raise TypeError
        self.dict_update_op(i)

    def list_append_op(self, i: int) -> None:
        val = self.pop()
        list.append(self.data_stack[-i], val)

    def map_add_op(self, i: int) -> None:
        val = self.pop()
        key = self.pop()
        dict.__setitem__(self.data_stack[-i], key, val)

    def build_string_op(self, count: int) -> None:
        self.push(''.join(self.pop_n(count)))

    def build_slice_op(self, argc: int) -> None:
        if argc == 2:
            tos = self.pop()
            tos1 = self.pop()
            self.push(slice(tos1, tos))
        elif argc == 3:
            tos = self.pop()
            tos1 = self.pop()
            tos2 = self.pop()
            self.push(slice(tos2, tos1, tos))
        else:
            raise ValueError("Invalid argc value")

    def list_to_tuple_op(self, arg: tp.Any) -> None:
        self.push(tuple(self.pop()))

    def nop_op(self, arg: tp.Any) -> None:
        pass

    def get_len_op(self) -> None:
        self.push(len(self.top()))

    def binary_op_op(self, op_code: int) -> None:
        x, y = self.pop_n(2)
        if op_code < 26:
            self.push(BINARY_OPERATORS[op_code](x, y))
        else:
            raise NotImplementedError(f"New binary op code: {op_code}")

    def compare_op_op(self, op_code: str) -> None:
        x, y = self.pop_n(2)
        if op_code in COMPARE_OPERATORS:
            self.push(COMPARE_OPERATORS[op_code](x, y))
        else:
            raise NotImplementedError(f"Unknown compare op code: {op_code}")

    def contains_op_op(self, invert: bool) -> None:
        x, y = self.pop_n(2)
        self.push((x not in y) if invert else (x in y))

    def is_op_op(self, invert: bool) -> None:
        x, y = self.pop_n(2)
        self.push((x is not y) if invert else (x is y))

    def unpack_sequence_op(self, count: int) -> None:
        seq = self.pop()
        for i in range(count):
            self.push(seq[~i])

    def format_value_op(self, arg: tp.Any) -> None:
        real_arg = self.instructions[self.instruction_pointer].arg
        if real_arg is None:
            return None
        specifier = self.pop() if real_arg & 4 else ""
        obj = self.pop()
        real_arg %= 4
        if real_arg == 0:
            self.push(f"{obj:{specifier}}")
        elif real_arg == 1:
            self.push(f"{obj!s:{specifier}}")
        elif real_arg == 2:
            self.push(f"{obj!r:{specifier}}")
        elif real_arg == 3:
            self.push(f"{obj!a:{specifier}}")

    def load_build_class_op(self, arg: tp.Any) -> None:
        self.push(self.builtins["__build_class__"])

    def setup_annotations_op(self, arg: tp.Any) -> None:
        if "__annotations__" not in self.locals:
            self.locals["__annotations__"] = {}

    def extended_arg_op(self, arg: tp.Any) -> None:
        pass


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'], globals_context, globals_context)
        return frame.run()
