from setuptools import find_packages,setup


def get_requirements(file_path):

    '''
    Fun reads the requirements.txt line by line 
    store in list, returns List with packages

    '''
    Hypo_e_dot = '-e .'
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

    if Hypo_e_dot in requirements:
        requirements.remove(Hypo_e_dot)
    
    return requirements



setup(
    name='mlproject',
    version='0.0.1',
    author='Jerrick',
    author_email='jerrick1105@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)