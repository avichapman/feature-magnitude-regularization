from typing import TypeVar, List

T = TypeVar("T")


class ArgumentHelper:
    """
    Ensures that values are of the expected type. Methods return the input variables so that they can be used inline.
    """

    @staticmethod
    def check_type(value: T, value_type: type) -> T:
        if not isinstance(value, value_type):
            raise ValueError('Value should be of type ' + str(value_type) + ". Found " + str(type(value)))

        return value

    @staticmethod
    def check_list_of_type(values: T, value_type: type) -> T:
        if not isinstance(values, List):
            raise ValueError('Value should be a list of of type ' + str(value_type) + ". Found " + str(type(values)))

        for item in values:
            if not isinstance(item, value_type):
                raise ValueError('Value should be of type ' + str(value_type)
                                 + ". Found element of type " + str(type(item)))

        return values

    @staticmethod
    def make_type(value: T, value_type: type) -> T:
        """
        Converts `value` to `value_type`. If that is not possible, raise an error.
        """
        if not isinstance(value, value_type):
            try:
                return value_type(value)
            except ValueError:
                raise ValueError('Value ' + str(value) + ' should castable to type ' +
                                 str(value_type) + ". Found " + str(type(value)))

        return value

    @staticmethod
    def check_type_or_none(value: T, value_type: type) -> T:
        if value is None:
            return value

        if not isinstance(value, value_type):
            raise ValueError('Value should be of type ' + str(value_type) + " or none. Found " + str(type(value)))

        return value
