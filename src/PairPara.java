
public class PairPara {

	private double labelScore;
	private double predictedScore;
	private double confScore;
	private String hypo;
	private String hyper;
	private boolean isTrain;

	public PairPara(double labelScore, double predictedScore, double confScore, String hypo, String hyper,
			boolean isTrain) {
		super();
		this.labelScore = labelScore;
		this.predictedScore = predictedScore;
		this.confScore = confScore;
		this.hypo = hypo;
		this.hyper = hyper;
		this.isTrain = isTrain;
	}

	public double getLabelScore() {
		return labelScore;
	}

	public void setLabelScore(double labelScore) {
		this.labelScore = labelScore;
	}

	public double getPredictedScore() {
		return predictedScore;
	}

	public void setPredictedScore(double predictedScore) {
		this.predictedScore = predictedScore;
	}

	public double getConfScore() {
		return confScore;
	}

	public void setConfScore(double confScore) {
		this.confScore = confScore;
	}

	public String getHypo() {
		return hypo;
	}

	public void setHypo(String hypo) {
		this.hypo = hypo;
	}

	public String getHyper() {
		return hyper;
	}

	public void setHyper(String hyper) {
		this.hyper = hyper;
	}

	public boolean isTrain() {
		return isTrain;
	}

	public void setTrain(boolean isTrain) {
		this.isTrain = isTrain;
	}

}
