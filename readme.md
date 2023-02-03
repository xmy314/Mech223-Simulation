sim result _ [density] _ [radius] _ [drag] _ [friction] _ [angular_property]

python -m cProfile -o simulation.profile simulation.py

snakeviz simulation.profile

