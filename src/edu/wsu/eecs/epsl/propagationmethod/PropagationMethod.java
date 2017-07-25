/**
 * author: Seyed Ali Rokni
 * 
 */
package edu.wsu.eecs.epsl.propagationmethod;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;
import java.util.Set;

import org.apache.commons.lang3.tuple.Pair;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.clusterers.SimpleKMeans;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import edu.wsu.eecs.epsl.SemiLabel;
import edu.wsu.eecs.epsl.Utils;

public class PropagationMethod {

	static Map<String, List<Double>> totals = new HashMap<String, List<Double>>();
	static Map<String, Classifier> classifiers = new HashMap<String, Classifier>();
	static String outputPrefix = "output\\KNear_";
	static String dataPrefix = "data\\";
	static {
		totals.put("Naiv", new ArrayList<>());
		totals.put("Ssup", new ArrayList<>());
		totals.put("Prop", new ArrayList<>());
		totals.put("Uppr", new ArrayList<>());
		totals.put("PNLA", new ArrayList<>());
		classifiers.put("KNN", new IBk(10));
		 classifiers.put("SVM", new LibSVM());
		classifiers.put("DT", new J48());
		 ((LibSVM) classifiers.get("SVM")).setKernelType(new
		 SelectedTag(LibSVM.KERNELTYPE_LINEAR,
		 LibSVM.TAGS_KERNELTYPE));
//		 // String[] options = {"-q"};
		 // ((LibSVM) classifiers.get("SVM")).setOptions(options);
		
		 libsvm.svm.svm_set_print_string_function(new
		 libsvm.svm_print_interface() {
		 @Override
		 public void print(String s) {
		 } // Disables svm output
		 });
	}

	public static void main(String[] args) throws Exception {
		 doResearch();
	}

	private static void generateConfusionMatrix() throws Exception {
		DataSet ds = DataSet.SENIOR;
		int iterationSize = 15;
		int part = 1;
		Set<Integer> trainSbj = new HashSet<>();
		Set<Integer> locations = new HashSet<>(Arrays.asList(1, 2, 3, 4, 5));
		trainSbj.addAll(ds.subjects);
		// Set<Integer> possibles = new HashSet<Integer>(trainSbj);
		Set<Integer> possibles = new HashSet<Integer>();
		possibles.add(3);
		int c = 0;
		for (int source = 1; source < 6; source++) {
			String fileName1 = outputPrefix + ds.toString() + "_source_" + source
					+ "_pre_confusion_2.txt";
			String fileName2 = outputPrefix + ds.toString() + "_source_" + source
					+ "_post_confusion_2.txt";

			FileWriter fw1 = new FileWriter(fileName1);
			FileWriter fw2 = new FileWriter(fileName2);
			locations.remove(source);
			double[][] confusionPre = new double[ds.numOfActivities][ds.numOfActivities];
			double[][] confusionPost = new double[ds.numOfActivities][ds.numOfActivities];
			for (int testSbj : possibles) {
				// perSource.put(testSbj, new ArrayList<>());

				trainSbj.remove(testSbj);
				// System.out.println("test: " + testSbj);
				Pair<Instances, Map<Integer, SemiLabel>> allTraining = getAllTraining(ds, trainSbj, testSbj, locations,
						source, part);

				Instances ins = removeLocation(allTraining.getLeft());
				Pair<Instances, List<Integer>> createForCluster = preProcessWithClasses(ins);
				List<Integer> groundTruthClasses = createForCluster.getRight();
				Instances train = createForCluster.getLeft();
				trainSbj.add(testSbj);

				Map<Integer, SemiLabel> indexToPredicted = allTraining.getRight();
				List<SemiGraphNode> graph = preprocessGraph(train, ds.k, indexToPredicted);
				SemiGraphNode.alpha = ds.alpha;
				SemiGraphNode.propagate(ds.k);

				for (int i = 0; i < graph.size(); i++) {
					confusionPre[groundTruthClasses.get(i)][indexToPredicted.get(i).predicted]++;
					confusionPost[groundTruthClasses.get(i)][graph.get(i).getClassValue()]++;
				}

				locations.add(source);
				System.out.println(++c);
			}
			fw1.write(Utils.matrixToString(confusionPre));
			fw2.write(Utils.matrixToString(confusionPost));
			fw1.close();
			fw2.close();

		}

	}

			
	private static Instances extraInstances(DataSet ds, int subject, int source, int part, int max) throws Exception {
		Instances result = null;
		Instances target = buildInstances(ds, subject, source, (part == 1 ? 2 : 1));
		Map<Integer, Integer> classCount = new HashMap<>();
		for (int i = 0; i < target.numInstances(); i++) {
			int c = (int) target.instance(i).classValue();
			if (!classCount.containsKey(c)) {
				if (result == null) {
					result = new Instances(target, i, 1);
				} else {
					result.add(target.instance(i));
				}
				classCount.put(c, 1);
			} else {
				Integer count = classCount.get(c);
				if (count < max) {
					result.add(target.instance(i));
					classCount.put(c, count + 1);
				}
			}
		}
		return result;
	}


/**
 * This method is an approximation/upper bound to Plug-n-Learn method. I does not perform matching.
 * Instead,it computes the best possible assignment to each cluster knowing labels. In other word,
 * Plug_n_Learn based on clustering cannot perform better than this.
 * @param ins
 * @param ds
 * @return
 * @throws Exception
 */
	private static Instances doPNLA(Instances ins, DataSet ds) throws Exception {
		Instances result = new Instances(ins);
		Instances ins2 = removeLocation(ins);
		ins2.setClassIndex(-1);
		SimpleKMeans kmeans = new SimpleKMeans();
		kmeans.setNumClusters(ds.numOfActivities);
		kmeans.setPreserveInstancesOrder(true);

		kmeans.buildClusterer(ins2);
		int[] assignment = kmeans.getAssignments();
		Map<Integer, List<Integer>> mapList = new HashMap<Integer, List<Integer>>();

		for (int i = 0; i < assignment.length; i++) {
			if (!mapList.containsKey(assignment[i])) {
				mapList.put(assignment[i], new ArrayList<>());
			}
			mapList.get(assignment[i]).add((int) ins.instance(i).classValue());
		}
		double allE = 0;
		int allN = 0;
		int e = 0;
		Map<Integer, Integer> newLabels = new HashMap<Integer, Integer>();
		for (Entry<Integer, List<Integer>> entry : mapList.entrySet()) {
			List<Integer> l = entry.getValue();
			int max = Collections.max(l);
			for (int i = 0; i < l.size(); i++) {
				if (l.get(i) != max) {
					e++;
				}
			}
			allN += l.size();

//			 System.out.println((double)e / l.size());
			newLabels.put(entry.getKey(), max);
		}
//		System.out.println((double) e / allN);
		for(int i = 0; i < ins.numInstances();i++){
			result.instance(i).setClassValue(newLabels.get(assignment[i]));
		}
		return result;
	}

	private static void doResearch() throws Exception {
		DataSet ds = DataSet.SAD;

		int part = 1;
		Set<Integer> trainSbj = new HashSet<>();
		Set<Integer> locations = new HashSet<>(Arrays.asList(1, 2, 3, 4, 5));
		trainSbj.addAll(ds.subjects);
		Set<Integer> possibles = new HashSet<Integer>(trainSbj);
		// Set<Integer> possibles = new HashSet<Integer>();
		// possibles.add(3);

		String fileName = outputPrefix + ds.toString() + "_result_test.txt";
		System.out.println(fileName);
		FileWriter fw = new FileWriter(fileName);
		int c = 0;
		for (int source = 1; source < 6; source++) {
			locations.remove(source);
			Map<String, ComparisonMethod> methods = new HashMap<String, ComparisonMethod>();
			for (String s : totals.keySet()) {
				methods.put(s, new ComparisonMethod(s));
			}
			List<Double> beforeLabels = new ArrayList<Double>();
			List<Double> afterLabels = new ArrayList<Double>();
			List<Double> pnlLabels = new ArrayList<Double>();
			for (int testSbj : possibles) {
				trainSbj.remove(testSbj);
				// System.out.println("test: " + testSbj);
				Pair<Instances, Map<Integer, SemiLabel>> allTraining = getAllTraining(ds, trainSbj, testSbj, locations,
						source, part);
				// trainSbj.remove(testSbj);
				Instances ins = removeLocation(allTraining.getLeft());
				/**
				 * For sake of comparison we create Four trains: Naive,
				 * system-supervised, afterPropagation and upperBound Also, we
				 * create the test instances
				 */
				Instances naive = buildGrouppedInstances(ds, trainSbj, source, part);
				naive = removeLocation(naive);
				Instances uppr = buildGrouppedInstances3(ds, trainSbj, testSbj, locations, part);
				uppr = removeLocation(uppr);
				// Instances uppr = new Instances(ins);
				// uppr.setClassIndex(uppr.numAttributes() - 1);
				Instances sSup = new Instances(ins);
				sSup.setClassIndex(sSup.numAttributes() - 1);
				Instances prop = new Instances(ins);
				prop.setClassIndex(prop.numAttributes() - 1);
				Instances pnla = doPNLA(ins, ds);
				pnla.setClassIndex(pnla.numAttributes() - 1);
				

				methods.get("Naiv").train = naive;
				methods.get("Ssup").train = sSup;
				methods.get("Prop").train = prop;
				methods.get("Uppr").train = uppr;
				methods.get("PNLA").train = pnla;

				Instances test = buildTestData(ds, testSbj, locations, part);
				test = removeLocation(test);
				/**
				 * End of comparison setup
				 */

				Pair<Instances, List<Integer>> createForCluster = preProcessWithClasses(ins);
				Instances train = createForCluster.getLeft();
				List<Integer> groundTruthClasses = createForCluster.getRight();
				trainSbj.add(testSbj);

				Map<Integer, SemiLabel> indexToPredicted = allTraining.getRight();
				List<SemiGraphNode> graph = preprocessGraph(train, ds.k, indexToPredicted);
				SemiGraphNode.alpha = ds.alpha;
				SemiGraphNode.propagate(ds.k);

				int missP = 0;
				int missG = 0;
				int missN = 0;
				for (int i = 0; i < graph.size(); i++) {
					prop.instance(i).setClassValue(graph.get(i).getClassValue());
					sSup.instance(i).setClassValue(indexToPredicted.get(i).predicted);
					if (groundTruthClasses.get(i) != indexToPredicted.get(i).predicted) {
						missP++;
					}
					if (groundTruthClasses.get(i) != graph.get(i).getClassValue()) {
						missG++;
					}
					if(groundTruthClasses.get(i) != pnla.instance(i).classValue()){
						missN++;
					}
				}
				beforeLabels.add((1 - (double) missP / graph.size()));
				afterLabels.add((1 - (double) missG / graph.size()));
				pnlLabels.add((1 - (double) missN / graph.size()));
				
				compareBeforeAndAfter(methods.values(), test);
				locations.add(source);
				System.out.println(++c);

			}
			fw.write(Utils.avgS(beforeLabels) + "\t" + Utils.avgS(afterLabels) + "\t" + Utils.avgS(pnlLabels));
			fw.write("\n");
			for (Entry<String, ComparisonMethod> m : methods.entrySet()) {
				fw.write(m.getKey() + " : " + m.getValue().toString() + "\n");
			}
			methods.clear();
		}
		fw.close();

	}

	private static List<Pair<Double, Integer>> computeKNearest(Instances train, int index, int k) {
		EuclideanDistance ed = new EuclideanDistance(train);
		List<Pair<Double, Integer>> distances = new ArrayList<Pair<Double, Integer>>();
		for (int i = 0; i < train.numInstances(); i++) {
			if (i != index) {
				distances.add(Pair.of(ed.distance(train.instance(index), train.instance(i)), i));
			}
		}
		Collections.sort(distances);
		return distances.subList(0, k);
	}

	private static List<SemiGraphNode> preprocessGraph(Instances train, int k, Map<Integer, SemiLabel> indexToPredicted)
			throws Exception {
		List<SemiGraphNode> graph = new ArrayList<SemiGraphNode>();
		for (int i = 0; i < train.numInstances(); i++) {
			List<Pair<Double, Integer>> kNearest = computeKNearest(train, i, k);
			List<Integer> indecies = new ArrayList<Integer>();
			for (Pair<Double, Integer> p : kNearest) {
				indecies.add(p.getRight());
			}
			SemiGraphNode sgn = new SemiGraphNode(indexToPredicted.get(i).possibles, indecies);
			graph.add(sgn);
		}
		SemiGraphNode.train = graph;
		return graph;
	}

	private static String buildFileName(DataSet dataset, int subject, Set<Integer> locations, int part) {
		switch (dataset) {
		case ETH:
			return dataPrefix + "OpportunityUCIDataset\\dataset\\eachLoc\\features\\S" + subject
					+ "-loc" + Utils.setToString(locations) + "_part_" + part + ".arff";
		case SENIOR:
			return dataPrefix + "surfaceV\\features\\sub" + subject + "-loc"
					+ Utils.setToString(locations) + "_part_" + part + ".arff";
		}
		return null;
	}

	private static String buildFileName(DataSet dataset, int subject, int location, int part) {
		switch (dataset) {
		case ETH:
			return dataPrefix + "OpportunityUCIDataset\\dataset\\eachLoc\\features\\S" + subject
					+ "-loc" + location + "_part_" + part + ".arff";
		case SAD:
			return dataPrefix + "SAD\\S" + subject + "-loc" + location + "_part_" + part
					+ ".arff";

		case SENIOR:
			return dataPrefix + "surfaceV\\features\\sub" + subject + "-loc" + location
					+ "_part_" + part + ".arff";
		}
		return null;
	}

	private static Instances buildGrouppedInstances(DataSet dataset, Set<Integer> trainSbj, Set<Integer> locations,
			int part) throws Exception {
		Instances train = null;
		for (int subject : trainSbj) {
			String buildFileName = buildFileName(dataset, subject, locations, part);
			// System.out.println(buildFileName);
			DataSource dataSource = new DataSource(buildFileName);
			if (train == null) {
				train = dataSource.getDataSet();
				train.setClassIndex(train.numAttributes() - 1);
			} else {
				Instances temp = dataSource.getDataSet();
				temp.setClassIndex(temp.numAttributes() - 1);
				for (int i = 0; i < temp.numInstances(); i++) {
					train.add(temp.instance(i));
				}
			}
		}
		if (dataset == DataSet.ETH) {
			train = doBalance(train);
		}
		return train;
	}

	private static Instances buildGrouppedInstances(DataSet dataset, Set<Integer> trainSbj, int location, int part)
			throws Exception {
		Instances train = null;
		for (int subject : trainSbj) {
			String fileName = buildFileName(dataset, subject, location, part);
			DataSource dataSource = new DataSource(fileName);
			// System.out.println(fileName);
			if (train == null) {
				train = dataSource.getDataSet();
				train.setClassIndex(train.numAttributes() - 1);
			} else {
				Instances temp = dataSource.getDataSet();
				temp.setClassIndex(temp.numAttributes() - 1);
				for (int i = 0; i < temp.numInstances(); i++) {
					train.add(temp.instance(i));
				}
			}
		}
		if (dataset == DataSet.ETH) {
			train = doBalance(train);
		}
		return train;
	}

	/**
	 * 
	 * 
	 * @param dataset
	 * @param trainSbj
	 * @param test
	 * @param locations
	 * @param part
	 * @return
	 * @throws Exception
	 */

	private static Instances buildGrouppedInstances3(DataSet dataset, Set<Integer> trainSbj, int test,
			Set<Integer> locations, int part) throws Exception {
		Instances train = null;
		trainSbj.add(test);
		for (int subject : trainSbj) {
			for (int location : locations) {
				String fileName = buildFileName(dataset, subject, location, (part == 1 ? 2 : 1));
				DataSource dataSource = new DataSource(fileName);
				// System.out.println(fileName);
				if (train == null) {
					train = dataSource.getDataSet();
					train.setClassIndex(train.numAttributes() - 1);
				} else {
					Instances temp = dataSource.getDataSet();
					temp.setClassIndex(temp.numAttributes() - 1);
					for (int i = 0; i < temp.numInstances(); i++) {
						train.add(temp.instance(i));
					}
				}
			}
		}
		if (dataset == DataSet.ETH) {
			train = doBalance(train);
		}
		return train;
	}

	private static Integer findMinClass(Instances train) {
		Map<Double, Integer> counts = new HashMap<Double, Integer>();
		for (int i = 0; i < train.numInstances(); i++) {
			double classValue = train.instance(i).classValue();
			if (!counts.containsKey(classValue)) {
				counts.put(classValue, 1);
			} else {
				counts.put(classValue, counts.get(classValue) + 1);
			}
		}
		int min = 100000;
		double minClass = 0;
		for (Entry<Double, Integer> e : counts.entrySet()) {
			if (min > e.getValue()) {
				min = e.getValue();
				minClass = e.getKey();
			}
		}
		return (int) minClass;
	}

	private static Instances doBalance(Instances train) throws Exception {
		return train;
		// SMOTE smote = new SMOTE();
		// smote.setClassValue("0");
		// smote.setNearestNeighbors(1);
		// smote.setPercentage(200);
		// smote.setInputFormat(train);
		// try{
		// train = Filter.useFilter(train, smote);
		// }catch(Exception ex){
		//
		// }
		// System.out.println(findMinClass(train));
		// Resample resample = new Resample();
		// resample.setBiasToUniformClass(1);
		// resample.setNoReplacement(false);
		// resample.setSampleSizePercent(100);
		// resample.setInputFormat(train);
		// findMinClass(train);
		// train = Filter.useFilter(train, resample);
		// System.out.println(findMinClass(train));
		// return train;

	}

	private static Instances buildInstances(DataSet dataset, int subject, int location, int part) throws Exception {
		DataSource dataSource = new DataSource(buildFileName(dataset, subject, location, part));
		Instances train = dataSource.getDataSet();
		train.setClassIndex(train.numAttributes() - 1);
		return train;
	}

	private static Instances buildInstances(DataSet dataset, int subject, Set<Integer> locations, int part)
			throws Exception {
		DataSource dataSource = new DataSource(buildFileName(dataset, subject, locations, part));
		Instances train = dataSource.getDataSet();
		train.setClassIndex(train.numAttributes() - 1);
		return train;
	}


	private static Pair<Instances, Map<Integer, SemiLabel>> getAllTraining(DataSet dataset, Set<Integer> trainSbj,
			int test, Set<Integer> locations, int source, int part) throws Exception {
		Map<Integer, SemiLabel> result = new HashMap<Integer, SemiLabel>();
		Instances train = buildGrouppedInstances(dataset, trainSbj, source, part);

		Instances extra = extraInstances(dataset, test, source, part, 2);
		for (int i = 0; i < extra.numInstances(); i++) {
			train.add(extra.instance(i));
		}

		Instances target = buildInstances(dataset, test, source, (part == 1 ? 2 : 1));
		Classifier ibk = new IBk(5);
		// ibk = classifiers.get("SVM");
		ibk.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(ibk, target);
		Map<Integer, SemiLabel> semiLabels = Utils.computeSemiLabels(eval.confusionMatrix());
		/**
		 * NEWLY ADDED
		 */
		// trainSbj.remove(test);
		// train = buildGrouppedInstances(dataset, trainSbj, source, part);
		// ibk.buildClassifier(train);
		Set<Integer> newTest = new HashSet<Integer>();
		newTest.add(test);
		newTest.add(trainSbj.iterator().next());
		/**
		 * End of NEWLY ADDED
		 */

		Instances newAllTraining = null;
		int index = 0;
		for (int loc : locations) {
			if (newAllTraining == null) {
				newAllTraining = buildInstances(dataset, test, loc, (part == 1 ? 2 : 1));
				/**
				 * NEWLY CHangED REDID
				 */
				// newAllTraining = buildGrouppedInstances(dataset, newTest,
				// loc, (part == 1 ? 2 : 1));
				newAllTraining.delete();
			}
			Instances perLocData = buildInstances(dataset, test, loc, (part == 1 ? 2 : 1));
			/**
			 * NEWLY CHangED - REDID
			 */
			// Instances perLocData = buildGrouppedInstances(dataset, newTest,
			// loc, (part == 1 ? 2 : 1));
			for (int i = 0; i < Math.min(target.numInstances(), perLocData.numInstances()); i++) {

				newAllTraining.add(perLocData.instance(i));
				int detectedClass = (int) ibk.classifyInstance(target.instance(i));
				result.put(index++, semiLabels.get(detectedClass));
			}
		}
		return Pair.of(newAllTraining, result);

	}

	private static Pair<Instances, Map<Integer, SemiLabel>> getCompoundAllTraining(DataSet dataset,
			Set<Integer> trainSbj, int test, Set<Integer> locations, Set<Integer> sources, int part) throws Exception {
		Map<Integer, SemiLabel> result = new HashMap<Integer, SemiLabel>();
		Instances train = buildGrouppedInstances(dataset, trainSbj, sources, part);
		Instances target = buildInstances(dataset, test, sources, (part == 1 ? 2 : 1));
		Classifier ibk = new IBk(3);
		// ibk = classifiers.get("SVM");
		ibk.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(ibk, target);
		System.out.println(eval.toSummaryString());
		Map<Integer, SemiLabel> semiLabels = Utils.computeSemiLabels(eval.confusionMatrix());
		Instances newAllTraining = null;
		int index = 0;
		for (int loc : locations) {
			if (newAllTraining == null) {
				newAllTraining = buildInstances(dataset, test, loc, (part == 1 ? 2 : 1));
				newAllTraining.delete();
			}
			Instances perLocData = buildInstances(dataset, test, loc, (part == 1 ? 2 : 1));
			for (int i = 0; i < Math.min(target.numInstances(), perLocData.numInstances()); i++) {

				newAllTraining.add(perLocData.instance(i));
				int detectedClass = (int) ibk.classifyInstance(target.instance(i));
				result.put(index++, semiLabels.get(detectedClass));
			}
		}
		return Pair.of(newAllTraining, result);

	}

	private static double getMeasure(char measure, Evaluation eval, int size) {
		if (measure == 'a') {
			return eval.pctCorrect();
		}
		// eval.
		double total = 0;
		for (int i = 0; i < size; i++) {
			switch (measure) {
			case 'p':
				total += eval.precision(i);
				break;
			case 'r':
				total += eval.recall(i);
				break;
			case 'f':
				total += eval.fMeasure(i);
				break;
			}
		}
		return total / size;
	}

	private static void computeMeasures(ComparisonMethod method, Instances test) throws Exception {
		Classifier classifier = classifiers.get("KNN");
		classifier.buildClassifier(method.train);
		Evaluation eval = new Evaluation(method.train);
		eval.evaluateModel(classifier, test);
		method.accuracies.add(eval.pctCorrect() / 100);
		method.percision.add(eval.weightedPrecision());
		method.recalls.add(eval.weightedRecall());
		method.fMeasures.add(eval.weightedFMeasure());

	}

	private static double getAccuracy(ComparisonMethod method, Instances test) throws Exception {
		Classifier classifier = classifiers.get("KNN");
		classifier.buildClassifier(method.train);
		Evaluation eval = new Evaluation(method.train);
		eval.evaluateModel(classifier, test);
		String result = Utils.formatter.format(eval.pctCorrect()) + "\t";
		method.output.append(result);
		return eval.pctCorrect();
	}

	private static void compareBeforeAndAfter(Collection<ComparisonMethod> methods, Instances test) throws Exception {
		for (ComparisonMethod method : methods) {
			try {
				// double accuracy = getAccuracy(method, test);
				// totals.get(method.name).add(accuracy);
				//
				computeMeasures(method, test);
			} catch (Exception e) {
				method.output.append("00.00 \t");
			}
		}
	}

	private static Instances buildTestData(DataSet dataset, int test, Set<Integer> locations, int part)
			throws Exception {
		Instances newAllTests = null;
		for (int loc : locations) {
			if (newAllTests == null) {
				newAllTests = buildInstances(dataset, test, loc, part);
				newAllTests.delete();
			}
			Instances perLocData = buildInstances(dataset, test, loc, part);
			for (int i = 0; i < perLocData.numInstances(); i++) {
				newAllTests.add(perLocData.instance(i));
			}
		}
		return newAllTests;
	}

	private static void testKNear() throws Exception {
		Set<Integer> train = new HashSet<>(Arrays.asList(16));
		Set<Integer> locations = new HashSet<>(Arrays.asList(2, 3, 4, 5));

		Pair<Instances, Map<Integer, SemiLabel>> allTraining = getAllTraining(DataSet.SENIOR, train, 18, locations, 1,
				1);
		Map<Integer, SemiLabel> indexToPredicted = allTraining.getRight();
		Pair<Instances, Pair<List<Integer>, List<Integer>>> result = Utils
				.preProcessWithClustersAndClasses(allTraining.getLeft());
		List<Integer> groundTruthLocations = result.getRight().getLeft();
		List<Integer> groundTruthClasses = result.getRight().getRight();
		for (int i = 0; i < result.getLeft().numInstances(); i++) {
			System.out.println("----------(" + groundTruthClasses.get(i) + ", " + groundTruthLocations.get(i)
					+ ")------");
			System.out.println("----------" + indexToPredicted.get(i));
			List<Pair<Double, Integer>> kNearests = computeKNearest(result.getLeft(), i, 5);
			for (Pair<Double, Integer> p : kNearests) {
				System.out.println("(" + groundTruthClasses.get(p.getRight()) + ", "
						+ groundTruthLocations.get(p.getRight()) + ")");
			}

		}
	}

	private static void resultExtractor() throws FileNotFoundException {
		Map<String, List<Double>> map = new HashMap<String, List<Double>>();
		for (int source = 1; source < 6; source++) {
			String fileName = outputPrefix + "ETH" + "_source" + source + "_alpha" + 0.5 + ".txt";
			Scanner sc = new Scanner(new File(fileName));
			sc.nextLine();
			sc.nextLine();
			for (int i = 0; i < 6; i++) {
				String line = sc.nextLine();
				String[] ss = line.split("\\t+");
				if (map.size() < 6) {
					map.put(ss[0], Arrays.asList(0d, 0d, 0d));// , 0d, 0d, 0d,
																// 0d, 0d, 0d));
				}
				// System.out.println(ss[9]);
				for (int j = 1; j < 4; j++) {
					double d = Double.parseDouble(ss[j].trim()) + map.get(ss[0]).get(j - 1);
					map.get(ss[0]).set(j - 1, d);
				}
			}
			// System.out.println(map);
			sc.close();
		}
		Map<Integer, StringBuffer> bufs = new HashMap<Integer, StringBuffer>();
		for (int i = 0; i < 3; i++) {
			bufs.put(i, new StringBuffer());
			bufs.get(i).append(Utils.formatter.format(map.get("Naiv: ").get(i) / 5) + " ");
			bufs.get(i).append(Utils.formatter.format(map.get("Ssup: ").get(i) / 5) + " ");
			bufs.get(i).append(Utils.formatter.format(map.get("Prop: ").get(i) / 5) + " ");
			bufs.get(i).append(Utils.formatter.format(map.get("Uppr: ").get(i) / 5) + " ");

		}
		for (Entry<Integer, StringBuffer> a : bufs.entrySet()) {
			System.out.print(a.getValue().toString() + ";");
		}
	}

	private static double getAllError(List<SemiGraphNode> list) {
		double total = 0;
		for (SemiGraphNode n : list) {
			total += n.getError();
		}
		return total;
	}

	private static Instances removeLocation(Instances ins) throws Exception {
		Remove rm = new Remove();
		Integer mote = (Integer) (ins.numAttributes()) - 1;
		rm.setAttributeIndices((mote).toString() + "-" + mote.toString());
		rm.setInputFormat(ins);
		ins = Filter.useFilter(ins, rm);
		ins.setClassIndex(ins.numAttributes() - 1);
		return ins;

	}

	private static Pair<Instances, List<Integer>> preProcessWithClasses(Instances instances) throws Exception {
		instances.setClassIndex(instances.numAttributes() - 1);
		List<Integer> classesRight = new ArrayList<Integer>();
		for (int i = 0; i < instances.numInstances(); i++) {
			classesRight.add((int) instances.instance(i).classValue());
		}

		Remove rm = new Remove();
		// Integer mote = (Integer) (instances.numAttributes());
		rm.setAttributeIndices("last");
		rm.setInputFormat(instances);
		instances = Filter.useFilter(instances, rm);
		instances.setClassIndex(-1);
		return Pair.of(instances, classesRight);

	}
}
