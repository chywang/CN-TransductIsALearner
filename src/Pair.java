
public class Pair implements Comparable<Pair> {
	private String entity;
	private double count;

	public Pair(String entity, double count) {
		super();
		this.entity = entity;
		this.count = count;
	}

	public String getEntity() {
		return entity;
	}

	public void setEntity(String entity) {
		this.entity = entity;
	}

	public double getCount() {
		return count;
	}

	public void setCount(double count) {
		this.count = count;
	}

	public int compareTo(Pair arg0) {
		if (count - arg0.count > 0)
			return 1;
		else if (count - arg0.count < 0)
			return -1;
		else
			return 0;
	}
}