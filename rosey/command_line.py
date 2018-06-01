def write_shell_script(command_string, output_filename='temp.sh', is_sudo=True):
    # Create
    with open(output_filename, 'w') as bash_file:
        bash_file.write('{}\n'.format(command_string))

    # Give shell script permission
    if is_sudo:
        run_process('sudo chmod +x {}'.format(output_filename))


def run_as_shell_script(command_string, output_filename='temp.sh', debug=False, is_sudo=True):
    import os

    write_shell_script(command_string, output_filename, is_sudo=is_sudo)

    # Run
    command = 'sudo bash' if is_sudo else 'bash'
    run_process('{} {}'.format(command, output_filename))

    # Destroy
    if debug is False:
        os.remove(output_filename)


def run_process(command_string: str):
    """
    Runs a command string and in process and waits for it to complete

    :param command_string:
    :return: was_successful -> boolean of whether it ran without error
    """
    import shlex
    import subprocess

    assert_is_linux()

    response = subprocess.call(shlex.split(command_string))
    was_successful = not bool(response)
    return was_successful


def run_parallel_processes(command_list: list, max_processes=-1):
    """
    Runs commands in a list in parallel and waits for them to complete

    :param max_processes:
    :param command_list:
    :return: list of commands that failed
    """
    import subprocess
    from tqdm import tqdm
    from functools import partial
    from multiprocessing import cpu_count
    from multiprocessing.dummy import Pool

    n = len(command_list)
    processor_pool_size = cpu_count() if n > cpu_count() else n
    processor_pool_size = max_processes if max_processes > 0 else processor_pool_size

    failed_commands = []
    pool = Pool(processor_pool_size)
    for i, response_code in tqdm(enumerate(pool.imap(partial(subprocess.call, shell=True), command_list)), total=n):
        if response_code != 0:
            failed_commands.append(command_list[i])

    return failed_commands


def string_to_bool(string):
    if str(string).lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif str(string).lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise ValueError('{} is not a boolean type'.format(string))


def generate_output_filename(filename: str, extension=None):
    """
    >>> generate_output_filename('test')
    'test_out'
    >>> generate_output_filename('test', extension='csv')
    'test_out.csv'
    >>> generate_output_filename('test.csv')
    'test_out'
    >>> generate_output_filename('test2.tsv.csv')
    'test2_out'
    >>> generate_output_filename('test.csv', extension='csv')
    'test_out.csv'
    """
    assert extension is None or '.' not in extension

    filename = filename.split('.', maxsplit=1)[0] + '_out'

    return filename if extension is None else filename + '.' + extension


def cprint_title(message: str):
    print()
    print('='*50)
    print(message)


def cprint(message: str):
    print('\n--- {} ---\n'.format(message))


def print_failed(messages):
    _ = [print('FAILED ->', x) for x in messages]


def print_list(iterable):
    _ = [print(i) for i in iterable]


def locate_files(file_pattern: str):
    """
    Exposes terminal's `locate` program from python. This is 1e5 times faster than python glob and os searching.
    NOTE: I know the business logic can be 1 line return statement but this will sacrifice readability for don't do it!

    :param file_pattern: String to find on the computer
    :return: list of results that can be empty.
    """
    import subprocess

    assert_is_linux()

    command = ['locate', file_pattern]
    output = subprocess.Popen(command, stdout=subprocess.PIPE).communicate()[0].decode()

    return output.split('\n')[:-1]


def extract_sample_ids_from_file_list(file_list):
    sample_ids = [sample_id[-6:] if sample_id.endswith(('D', 'L', 'T')) else sample_id[-4:] for sample_id in file_list]
    return [ids for ids in sample_ids if ids[:4].isdigit()]


def directory_files(pattern=None) -> list:
    """
    Gets all the files names in the current working directory or all the files that match a particular pattern

    :param pattern: List to check if the file name contains
    :return: list of files
    """
    import os

    def get_all_files():
        return [x for x in os.listdir()]

    assert pattern is None or type(pattern) is str or type(pattern) is list, 'Must be list, None or str got {}'.format(type(pattern))

    output_filenames = []

    if pattern is None:
        output_filenames = get_all_files()
    elif type(pattern) is str:
        output_filenames = [x for x in os.listdir() if pattern in x]
    else:
        if len(pattern) > 0:
            output_filenames = get_all_files()
            for i_pattern in pattern:
                output_filenames = [x for x in output_filenames if i_pattern in x]

    return output_filenames


def get_first_item_else_none(array):
    return array[0] if len(array) > 0 else None


def count_lines_in_file(file_path) -> int:
    with open(file_path) as file:
        n_lines = sum(1 for _ in file)
    return n_lines


def is_linux():
    from sys import platform
    return 'linux' in platform


def assert_is_linux():
    from rosey.Errors import NotUbuntuError
    if not is_linux():
        raise NotUbuntuError('This must run on Ubuntu')


def assert_valid_python_version():
    import sys
    assert sys.version_info.major == 3, 'This script must be run on Python 3'


def limit_openblas_threads(n_threads: int=1):
    import os
    if is_linux():
        os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)


def validate_running_credentials(user_name='beesy'):
    import getpass
    assert getpass.getuser() == user_name, 'This script must be run by `{}`'.format(user_name)
    assert_valid_python_version()


def split_csv(filename, n_splits=-1, verbose=False):
    """
    Splits a file with a header to N files and returns the list of split file names

    Uses bash commands for speed

    :param filename:
    :param n_splits:
    :param verbose: Print progress through the process
    :return:
    """
    import math
    from multiprocessing import cpu_count
    assert_is_linux()

    if n_splits == -1:
        n_splits = cpu_count()

    # Split PVCF into even parts
    if verbose:
        print(f'Solving split sizes...')
    n_lines = count_lines_in_file(filename)
    lines_per_part = math.ceil(n_lines / n_splits)

    if verbose:
        print(f'Splitting {filename}')
    split_cmd = f'split --lines {lines_per_part} {filename} {filename + ".temp."}'
    run_as_shell_script(split_cmd, f'split-{filename}.sh', is_sudo=False)
    subfile_list = directory_files([filename, '.temp.'])
    subfile_list.sort()

    # Add headers to sub files
    add_headers_cmd = f'head -n 1 {filename} > header_file\n'
    for part in subfile_list[1:]:
        temp_part = part + '.ephemeral'

        add_headers_cmd += f'cat header_file {part} > {temp_part}\n'
        add_headers_cmd += f'mv {temp_part} {part}\n'
    add_headers_cmd += 'rm header_file'
    run_as_shell_script(add_headers_cmd, f'add-headers-{filename}.sh', is_sudo=False)

    if len(subfile_list) != n_splits:
        raise AssertionError(f'{filename} did not split correctly')

    return subfile_list


if __name__ == '__main__':
    import doctest
    doctest.testmod()
