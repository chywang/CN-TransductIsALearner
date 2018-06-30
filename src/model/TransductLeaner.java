package model;

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

import com.ansj.vec.Word2VEC;

import Jama.Matrix;
import edu.fudan.nlp.cn.tag.POSTagger;

public class TransductLeaner {

	private final static String word2vecPath="/Users/bear/Documents/workspace/Hearst/javaSkip50.model";
	private final static String trainPath="train.txt";
	private final static String testPath="output.txt";
	private final static String blackListPath="blacklist.txt";
	private final static String transductPath="transduct_output.txt";
	
	private Map<String, List<PairPara>> map=new HashMap<String, List<PairPara>>();
	private Word2VEC vec = new Word2VEC();
	private POSTagger tag;
	private Set<String> blacklist=new HashSet<String>();
	
	public TransductLeaner() throws Exception {
		vec.loadJavaModel(word2vecPath);
		tag = new POSTagger("fdnlp/seg.m","fdnlp/pos.m");
		System.out.println("model load");
		loadBlackList();
		loadTrain();
		loadTest();
	}
	
	private void loadTrain() throws IOException {
		BufferedReader br=new BufferedReader(new FileReader(new File(trainPath)));
		String line;
		while ((line=br.readLine())!=null) {
			String[] items=line.split("\t");
			String hypo=items[0];
			String hyper=items[1];
			String label=items[2];
			if (label.equals("1")) {
				//positive
				PairPara pairPara=new PairPara(1, 1, 1, hypo, hyper, true);
				if (!map.containsKey(hyper))
					map.put(hyper, new ArrayList<PairPara>());
				List<PairPara> pairParas=map.get(hyper);
				pairParas.add(pairPara);
				map.put(hyper, pairParas);
			} else if (label.equals("0")) {
				//negative
				PairPara pairPara=new PairPara(-1, -1, 1, hypo, hyper, true);
				if (!map.containsKey(hyper))
					map.put(hyper, new ArrayList<PairPara>());
				List<PairPara> pairParas=map.get(hyper);
				pairParas.add(pairPara);
				map.put(hyper, pairParas);
			}
		}
		br.close();
	}
	
	private void loadBlackList() throws IOException {
		BufferedReader br=new BufferedReader(new FileReader(new File(blackListPath)));
		String line;
		while ((line=br.readLine())!=null) {
			blacklist.add(line);
		}
		br.close();
	}
			
	private void loadTest() throws IOException {
		BufferedReader br=new BufferedReader(new FileReader(new File(testPath)));
		String line;
		while ((line=br.readLine())!=null) {
			String[] items=line.split("\t");
			String hypo=items[0];
			String hyper=items[1];
			double label=Double.parseDouble(items[2]);
			if (label==0)
				label=-1;
			double predicted=Double.parseDouble(items[3]);
			double conf=Double.parseDouble(items[4]);
			PairPara pairPara=new PairPara(label, predicted, conf, hypo, hyper, false);
			if (!map.containsKey(hyper))
				map.put(hyper, new ArrayList<PairPara>());
			List<PairPara> pairParas=map.get(hyper);
			pairParas.add(pairPara);
			map.put(hyper, pairParas);
		}
		br.close();
	}
	
	private void learnModel() throws IOException {
		for (String hyper:map.keySet()) {
			learnForHyper(hyper);
		}
	}
	
	public void printResultToFile() throws IOException {
		List<String> pairs=new ArrayList<String>();
		for (String s:map.keySet()) {
			List<PairPara> list=map.get(s);
			for (PairPara pairPara:list) {
				if (pairPara.isTrain())
					continue;
				pairs.add(pairPara.getHypo()+"\t"+pairPara.getHyper()+"\t"+pairPara.getLabelScore()+"\t"+pairPara.getPredictedScore());
			}
		}
		Collections.sort(pairs);
		PrintWriter pw=new PrintWriter(transductPath);
		for (String p:pairs) {
			pw.println(p);
			pw.flush();
		}
		pw.close();
	}
	
	private void learnForHyper(String hyper) {
		List<PairPara> pairParas=map.get(hyper);
		System.out.println(pairParas.size());
		if (pairParas.size()==1)
			return;
		Matrix F=getInitialization(pairParas);
		Matrix S=getS(pairParas);
		Matrix Ws=getWS(pairParas);
		Matrix sigma=getSigma(pairParas);
		Matrix sigmaInv=sigma.inverse();
		
		//parameters
		double gamma=0.0001;
		double mu=0.0001;
		double eta=0.000000001;
		while (true) {
			Matrix cMatrix=getC(pairParas);
			Matrix changeF=Ws.times(Ws).times(F.minus(S)).plus(F.minus(cMatrix)).plus(sigmaInv.times(F).times(gamma)).plus(F.times(mu));
			Matrix newF=F.minus(changeF.times(eta));	
			System.out.println(newF.minus(F).normF());
			if (newF.minus(F).normF()<0.001) {
				F=newF;
				break;
			} else
				F=newF;
		}
		List<PairPara> list=map.get(hyper);
		for (int i=0;i<list.size();i++) {
			PairPara pairPara=list.get(i);
			pairPara.setPredictedScore(F.get(i, 0));
			list.set(i, pairPara);
		}
		map.put(hyper, list);
		System.out.println("train done");
	}
	
	private Matrix getS(List<PairPara> pairParas) {
		double[][] d=new double[pairParas.size()][1];
		for (int i=0;i<pairParas.size();i++) {
			PairPara pairPara=pairParas.get(i);
			d[i][0]=pairPara.getPredictedScore();
		}
		Matrix matrix=new Matrix(d);
		return matrix;
	}
	
	private Matrix getInitialization(List<PairPara> pairParas) {
		double[][] d=new double[pairParas.size()][1];
		for (int i=0;i<pairParas.size();i++) {
			PairPara pairPara=pairParas.get(i);
			d[i][0]=pairPara.getPredictedScore();
		}
		Matrix matrix=new Matrix(d);
		return matrix;
	}
	
	private Matrix getSigma(List<PairPara> pairParas) {
		double[][] d=new double[pairParas.size()][pairParas.size()];
		for (int i=0;i<pairParas.size();i++) {
			for (int j=0;j<pairParas.size();j++) {
				if (i==j)
					d[i][i]=1;
				else {
					String first=pairParas.get(i).getHypo();
					String second=pairParas.get(j).getHypo();
					float[] x=vec.getWordVector(first);
					float[] y=vec.getWordVector(second);
					double cos=cosine(x, y);
					d[i][j]=cos;
					d[j][i]=cos;
				}	
			}
		}
		Matrix matrix=new Matrix(d);
		return matrix;
	}
	
	private Matrix getWS(List<PairPara> pairParas) {
		double[][] d=new double[pairParas.size()][pairParas.size()];
		for (int i=0;i<pairParas.size();i++) {
			PairPara pairPara=pairParas.get(i);
			d[i][i]=pairPara.getConfScore();
		}
		Matrix matrix=new Matrix(d);
		return matrix;
	}
	
	@SuppressWarnings("unused")
	private void printMatrix(Matrix m) {
		double[][] d=m.getArray();
		for (int i=0;i<d.length;i++) {
			for (int j=0;j<d[i].length;j++) {
				System.out.print(d[i][j]+" ");
			}
			System.out.println();
		}
	}

	private double cosine(float[] x, float[] y) {
		double product=0;
		for (int i=0;i<x.length;i++)
			product+=x[i]*y[i];
		return product/(norm(x)*norm(y));
	}
	
	private double norm(float[] x) {
		double square=0;
		for (int i=0;i<x.length;i++)
			square+=x[i]*x[i];
		return Math.sqrt(square);
	}
	
	
	private Matrix getC(List<PairPara> pairParas) {
		double[][] d=new double[pairParas.size()][1];
		for (int i=0;i<pairParas.size();i++) {
			PairPara pairPara=pairParas.get(i);
			if (pairPara.isTrain())
				d[i][0]=pairPara.getLabelScore();
			else {
				String hyper=pairPara.getHyper();
				String hypo=pairPara.getHypo();
				if (hyper.endsWith(hypo)) {
					if (pairPara.getPredictedScore()<0.967)
						d[i][0]=0.976;
					else
						d[i][0]=pairPara.getPredictedScore();
				} else if (hyper.indexOf(getHead(hypo))>=0) {
					if (pairPara.getPredictedScore()>-0.968)
						d[i][0]=-0.968;
					else
						d[i][0]=pairPara.getPredictedScore();
				} else if (blacklist.contains(getHead(hyper))) {
					if (pairPara.getPredictedScore()>-0.973)
						d[i][0]=-0.973;
					else
						d[i][0]=pairPara.getPredictedScore();
				}
				d[i][0]=pairPara.getPredictedScore();
			}
		}
		Matrix matrix=new Matrix(d);
		return matrix;
	}
	
	private String getHead(String input)  {
		String[] items=tag.tag(input).split(" ");
		String keyword=items[items.length-1];
		String word=keyword.substring(0, keyword.indexOf("/"));
		return word;
	}
	
	
	public static void main(String[] args) throws Exception {
		TransductLeaner leaner=new TransductLeaner();
		leaner.learnModel();
		leaner.printResultToFile();
	}

}
