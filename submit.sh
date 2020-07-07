for compgoal in 0.3 0.4 0.5;
	do for i in {1..8};
		do sbatch runner1.sh $i $compgoal
	done
done
