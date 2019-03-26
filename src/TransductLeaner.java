
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import Jama.Matrix;
import edu.fudan.nlp.cn.tag.POSTagger;

public class TransductLeaner {

	private String w2vPath = "word_vectors.txt";
	private String trainPath = "train.txt";
	private String testPath = "test.txt";
	private final String initialPath = "initial.txt";
	private String outputPath = "output.txt";
	private final String blackListPath = "blacklist.txt";

	private int dimension = 50;
	private final static double lambda = 0.001;

	private Map<String, List<PairPara>> map = new HashMap<String, List<PairPara>>();
	private Map<String, float[]> vectors = new HashMap<String, float[]>();
	private POSTagger tag;
	private Set<String> blacklist = new HashSet<String>();

	public TransductLeaner(String w2vPath, String trainPath, String testPath, String outputPath, int dimension)
			throws Exception {
		this.w2vPath = w2vPath;
		this.trainPath = trainPath;
		this.testPath = testPath;
		this.outputPath = testPath;
		this.dimension = dimension;

		loadW2V();
		tag = new POSTagger("fdnlp/seg.m", "fdnlp/pos.m");
		System.out.println("software initialized!");
	}

	public void initialPredict() throws IOException {
		Matrix trainHypoIsA = loadHypoMatrix(trainPath, true);
		Matrix trainHyperIsA = loadHyperMatrix(trainPath, true);
		Matrix MIsA = computeM(trainHypoIsA, trainHyperIsA, lambda);
		Matrix trainHypoNotIsA = loadHypoMatrix(trainPath, false);
		Matrix trainHyperNotIsA = loadHyperMatrix(trainPath, false);
		Matrix MNotIsA = computeM(trainHypoNotIsA, trainHyperNotIsA, lambda);
		predict(testPath, initialPath, MIsA, MNotIsA);
	}

	private void predict(String testPath, String initialPath, Matrix MIsA, Matrix MNotIsA) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(testPath)));
		PrintWriter pw = new PrintWriter(initialPath);
		String line;
		while ((line = br.readLine()) != null) {
			String[] items = line.split("\t");
			String hypo = items[0];
			String hyper = items[1];
			pw.println(hypo + "\t" + hyper + "\t" + computeScore(hypo, hyper, MIsA, MNotIsA) + "\t"
					+ computeConf(hypo, hyper, MIsA, MNotIsA));
			pw.flush();
		}
		br.close();
		pw.close();
	}

	private double computeScore(String hypo, String hyper, Matrix MIsA, Matrix MNotIsA) {
		float[] hf1 = vectors.get(hypo);
		double[][] ds1 = new double[hf1.length][1];
		for (int i = 0; i < hf1.length; i++)
			ds1[i][0] = hf1[i];
		Matrix hypoM = new Matrix(ds1);
		float[] hf2 = vectors.get(hyper);
		double[][] ds2 = new double[hf2.length][1];
		for (int i = 0; i < hf2.length; i++)
			ds2[i][0] = hf2[i];
		Matrix hyperM = new Matrix(ds2);
		double positiveScore = MIsA.times(hypoM).minus(hyperM).normF();
		double negativeScore = MNotIsA.times(hypoM).minus(hyperM).normF();
		return Math.tanh(negativeScore - positiveScore);
	}

	private double computeConf(String hypo, String hyper, Matrix MIsA, Matrix MNotIsA) {
		float[] hf1 = vectors.get(hypo);
		double[][] ds1 = new double[hf1.length][1];
		for (int i = 0; i < hf1.length; i++)
			ds1[i][0] = hf1[i];
		Matrix hypoM = new Matrix(ds1);
		float[] hf2 = vectors.get(hyper);
		double[][] ds2 = new double[hf2.length][1];
		for (int i = 0; i < hf2.length; i++)
			ds2[i][0] = hf2[i];
		Matrix hyperM = new Matrix(ds2);
		double positiveScore = MIsA.times(hypoM).minus(hyperM).normF();
		double negativeScore = MNotIsA.times(hypoM).minus(hyperM).normF();
		return Math.abs(negativeScore - positiveScore) / Math.max(positiveScore, negativeScore);
	}

	private Matrix loadHypoMatrix(String path, boolean isPositive) throws IOException {
		List<String> words = new ArrayList<String>();
		BufferedReader br = new BufferedReader(new FileReader(new File(path)));
		String line;
		while ((line = br.readLine()) != null) {
			String[] items = line.split("\t");
			String hypo = items[0];
			String label = items[2];
			if (isPositive) {
				if (label.equals("1"))
					words.add(hypo);
			} else {
				if (label.equals("0"))
					words.add(hypo);
			}
		}
		br.close();

		double[][] ds1 = new double[dimension][words.size()];
		for (int m = 0; m < words.size(); m++) {
			float[] hf1 = vectors.get(words.get(m));
			for (int i = 0; i < hf1.length; i++)
				ds1[i][m] = hf1[i];
		}
		br.close();
		return new Matrix(ds1);
	}

	private Matrix loadHyperMatrix(String path, boolean isPositive) throws IOException {
		List<String> words = new ArrayList<String>();
		BufferedReader br = new BufferedReader(new FileReader(new File(path)));
		String line;
		while ((line = br.readLine()) != null) {
			String[] items = line.split("\t");
			String hyper = items[1];
			String label = items[2];
			if (isPositive) {
				if (label.equals("1"))
					words.add(hyper);
			} else {
				if (label.equals("0"))
					words.add(hyper);
			}
		}
		br.close();
		double[][] ds1 = new double[dimension][words.size()];
		for (int m = 0; m < words.size(); m++) {
			float[] hf1 = vectors.get(words.get(m));
			for (int i = 0; i < hf1.length; i++)
				ds1[i][m] = hf1[i];
		}
		br.close();
		return new Matrix(ds1);
	}

	private Matrix computeM(Matrix xMatrix, Matrix yMatrix, double lambda) {
		double[][] d = new double[dimension][dimension];
		for (int i = 0; i < d.length; i++)
			for (int j = 0; j < d[i].length; j++) {
				if (i == j)
					d[i][j] = 1;
				else
					d[i][j] = 0;
			}
		Matrix identity = new Matrix(d);
		return yMatrix.times(xMatrix.transpose())
				.times((xMatrix.times(xMatrix.transpose()).plus(identity.times(lambda))).inverse());
	}

	private void loadW2V() throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(w2vPath)));
		String line;
		while ((line = br.readLine()) != null) {
			String[] items = line.split(" ");
			String word = items[0];
			float[] vector = new float[items.length - 1];
			for (int i = 0; i < vector.length; i++)
				vector[i] = Float.parseFloat(items[i + 1]);
			vectors.put(word, vector);
		}
		br.close();
	}

	private void loadTrainingSet() throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(trainPath)));
		String line;
		while ((line = br.readLine()) != null) {
			String[] items = line.split("\t");
			String hypo = items[0];
			String hyper = items[1];
			String label = items[2];
			if (label.equals("1")) {
				// positive
				PairPara pairPara = new PairPara(1, 1, 1, hypo, hyper, true);
				if (!map.containsKey(hyper))
					map.put(hyper, new ArrayList<PairPara>());
				List<PairPara> pairParas = map.get(hyper);
				pairParas.add(pairPara);
				map.put(hyper, pairParas);
			} else if (label.equals("0")) {
				// negative
				PairPara pairPara = new PairPara(-1, -1, 1, hypo, hyper, true);
				if (!map.containsKey(hyper))
					map.put(hyper, new ArrayList<PairPara>());
				List<PairPara> pairParas = map.get(hyper);
				pairParas.add(pairPara);
				map.put(hyper, pairParas);
			}
		}
		br.close();
	}

	private void loadBlackList() throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(blackListPath)));
		String line;
		while ((line = br.readLine()) != null) {
			blacklist.add(line);
		}
		br.close();
	}

	private void loadInitialPredictions() throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(initialPath)));
		String line;
		while ((line = br.readLine()) != null) {
			String[] items = line.split("\t");
			String hypo = items[0];
			String hyper = items[1];
			double predicted = Double.parseDouble(items[2]);
			double conf = Double.parseDouble(items[3]);
			PairPara pairPara = new PairPara(predicted, predicted, conf, hypo, hyper, false);
			if (!map.containsKey(hyper))
				map.put(hyper, new ArrayList<PairPara>());
			List<PairPara> pairParas = map.get(hyper);
			pairParas.add(pairPara);
			map.put(hyper, pairParas);
		}
		br.close();
	}

	private void tranductiveLearn() throws IOException {
		loadBlackList();
		loadTrainingSet();
		loadInitialPredictions();
		for (String hyper : map.keySet()) {
			learnForHyper(hyper);
		}
		printResultToFile();
	}

	private void printResultToFile() throws IOException {
		List<String> pairs = new ArrayList<String>();
		for (String s : map.keySet()) {
			List<PairPara> list = map.get(s);
			for (PairPara pairPara : list) {
				if (pairPara.isTrain())
					continue;
				pairs.add(pairPara.getHypo() + "\t" + pairPara.getHyper() + "\t" + pairPara.getPredictedScore());
			}
		}
		Collections.sort(pairs);
		PrintWriter pw = new PrintWriter(outputPath);
		for (String p : pairs) {
			pw.println(p);
			pw.flush();
		}
		pw.close();
	}

	private void learnForHyper(String hyper) {
		List<PairPara> pairParas = map.get(hyper);
		System.out.println(pairParas.size());
		if (pairParas.size() == 1)
			return;
		Matrix F = getInitialization(pairParas);
		Matrix S = getS(pairParas);
		Matrix Ws = getWS(pairParas);
		Matrix sigma = getSigma(pairParas);
		Matrix sigmaInv = sigma.inverse();

		// parameters
		double gamma = 0.0001;
		double mu = 0.0001;
		double eta = 0.000000001;
		while (true) {
			Matrix cMatrix = getC(pairParas);
			Matrix changeF = Ws.times(Ws).times(F.minus(S)).plus(F.minus(cMatrix)).plus(sigmaInv.times(F).times(gamma))
					.plus(F.times(mu));
			Matrix newF = F.minus(changeF.times(eta));
			System.out.println(newF.minus(F).normF());
			if (newF.minus(F).normF() < 0.001) {
				F = newF;
				break;
			} else
				F = newF;
		}
		List<PairPara> list = map.get(hyper);
		for (int i = 0; i < list.size(); i++) {
			PairPara pairPara = list.get(i);
			pairPara.setPredictedScore(F.get(i, 0));
			list.set(i, pairPara);
		}
		map.put(hyper, list);
		System.out.println("train done");
	}

	private Matrix getS(List<PairPara> pairParas) {
		double[][] d = new double[pairParas.size()][1];
		for (int i = 0; i < pairParas.size(); i++) {
			PairPara pairPara = pairParas.get(i);
			d[i][0] = pairPara.getPredictedScore();
		}
		Matrix matrix = new Matrix(d);
		return matrix;
	}

	private Matrix getInitialization(List<PairPara> pairParas) {
		double[][] d = new double[pairParas.size()][1];
		for (int i = 0; i < pairParas.size(); i++) {
			PairPara pairPara = pairParas.get(i);
			d[i][0] = pairPara.getPredictedScore();
		}
		Matrix matrix = new Matrix(d);
		return matrix;
	}

	private Matrix getSigma(List<PairPara> pairParas) {
		double[][] d = new double[pairParas.size()][pairParas.size()];
		for (int i = 0; i < pairParas.size(); i++) {
			for (int j = 0; j < pairParas.size(); j++) {
				if (i == j)
					d[i][i] = 1;
				else {
					String first = pairParas.get(i).getHypo();
					String second = pairParas.get(j).getHypo();
					float[] x = vectors.get(first);
					float[] y = vectors.get(second);
					double cos = cosine(x, y);
					d[i][j] = cos;
					d[j][i] = cos;
				}
			}
		}
		Matrix matrix = new Matrix(d);
		return matrix;
	}

	private Matrix getWS(List<PairPara> pairParas) {
		double[][] d = new double[pairParas.size()][pairParas.size()];
		for (int i = 0; i < pairParas.size(); i++) {
			PairPara pairPara = pairParas.get(i);
			d[i][i] = pairPara.getConfScore();
		}
		Matrix matrix = new Matrix(d);
		return matrix;
	}

	@SuppressWarnings("unused")
	private void printMatrix(Matrix m) {
		double[][] d = m.getArray();
		for (int i = 0; i < d.length; i++) {
			for (int j = 0; j < d[i].length; j++) {
				System.out.print(d[i][j] + " ");
			}
			System.out.println();
		}
	}

	private double cosine(float[] x, float[] y) {
		double product = 0;
		for (int i = 0; i < x.length; i++)
			product += x[i] * y[i];
		return product / (norm(x) * norm(y));
	}

	private double norm(float[] x) {
		double square = 0;
		for (int i = 0; i < x.length; i++)
			square += x[i] * x[i];
		return Math.sqrt(square);
	}

	private Matrix getC(List<PairPara> pairParas) {
		double[][] d = new double[pairParas.size()][1];
		for (int i = 0; i < pairParas.size(); i++) {
			PairPara pairPara = pairParas.get(i);
			if (pairPara.isTrain())
				d[i][0] = pairPara.getLabelScore();
			else {
				String hyper = pairPara.getHyper();
				String hypo = pairPara.getHypo();
				if (hyper.endsWith(hypo)) {
					if (pairPara.getPredictedScore() < 0.967)
						d[i][0] = 0.976;
					else
						d[i][0] = pairPara.getPredictedScore();
				} else if (hyper.indexOf(getHead(hypo)) >= 0) {
					if (pairPara.getPredictedScore() > -0.968)
						d[i][0] = -0.968;
					else
						d[i][0] = pairPara.getPredictedScore();
				} else if (blacklist.contains(getHead(hyper))) {
					if (pairPara.getPredictedScore() > -0.973)
						d[i][0] = -0.973;
					else
						d[i][0] = pairPara.getPredictedScore();
				}
				d[i][0] = pairPara.getPredictedScore();
			}
		}
		Matrix matrix = new Matrix(d);
		return matrix;
	}

	private String getHead(String input) {
		String[] items = tag.tag(input).split(" ");
		String keyword = items[items.length - 1];
		String word = keyword.substring(0, keyword.indexOf("/"));
		return word;
	}

	public static void main(String[] args) throws Exception {
		if (args.length != 5) {
			System.out.println("argments input error!");
			return;
		}
		
		String w2vPath=args[0];
		String trainPath=args[1];
		String testPath = args[2];
		String outputPath = args[3];
		int dimension = Integer.parseInt(args[4]);
		
		TransductLeaner leaner = new TransductLeaner(w2vPath, trainPath, testPath, outputPath, dimension);
		leaner.initialPredict();
		leaner.tranductiveLearn();
	}

}
