from setuptools import setup

package_name = 'robot_model_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mausumi Bhuyan',
    maintainer_email='your_email@example.com',
    description='Wheeled robot package',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'random_mover = robot_model_pkg.random_mover.random_mover:main',
        ],
    },
)
