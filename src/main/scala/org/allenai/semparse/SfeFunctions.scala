package org.allenai.semparse

import java.util.{List => JList}

import scala.collection.JavaConverters._

import com.google.common.base.Preconditions

import com.jayantkrish.jklol.lisp.AmbEval.AmbFunctionValue
import com.jayantkrish.jklol.lisp.ConstantValue
import com.jayantkrish.jklol.lisp.ConsValue
import com.jayantkrish.jklol.lisp.{Environment => JEnv}
import com.jayantkrish.jklol.lisp.ParametricBfgBuilder
import com.jayantkrish.jklol.lisp.SpecAndParameters

class CreateSfeFeatureComputer extends AmbFunctionValue {
  override def apply(argumentValues: JList[Object], env: JEnv, b: ParametricBfgBuilder) = {
    Preconditions.checkArgument(argumentValues.size() == 1)
    val specFile = argumentValues.get(0).asInstanceOf[String]
    new SfeFeatureComputer(specFile)
  }
}

class DisplayParameters extends AmbFunctionValue {
  override def apply(argumentValues: JList[Object], env: JEnv, b: ParametricBfgBuilder) = {
    Preconditions.checkArgument(argumentValues.size() == 1)
    val params = argumentValues.get(0).asInstanceOf[SpecAndParameters]
    val stats = params.getParameters()
    println(stats.getDescription())
    ConstantValue.TRUE
  }
}

class FindRelatedEntities extends AmbFunctionValue {
  override def apply(argumentValues: JList[Object], env: JEnv, b: ParametricBfgBuilder) = {
    Preconditions.checkArgument(argumentValues.size() == 2)
    val midRelations = ConsValue.consListToList(argumentValues.get(0))
    val featureComputer = argumentValues.get(1).asInstanceOf[SfeFeatureComputer]

    val parsedMidRelations = midRelations.asScala.map(midRelation => {
      val list = ConsValue.consListToList(midRelation).asScala.toSeq
      val word = list(0).asInstanceOf[String]
      val mid = list(1).asInstanceOf[String]
      val isSource = list(2).asInstanceOf[Boolean]
      (word, mid, isSource)
    })

    val relatedEntities = parsedMidRelations.flatMap {
      case (word, mid, isSource) => featureComputer.findRelatedEntities(word, mid, isSource)
    }
    relatedEntities.toArray
  }
}

class GetCatWordFeatureList extends AmbFunctionValue {
  override def apply(argumentValues: JList[Object], env: JEnv, b: ParametricBfgBuilder) = {
    Preconditions.checkArgument(argumentValues.size() == 2)
    val word = argumentValues.get(0).asInstanceOf[String]
    val featureComputer = argumentValues.get(1).asInstanceOf[SfeFeatureComputer]

    val features = featureComputer.getFeaturesForCatWord(word).asJava
    ConsValue.listToConsList(features)
  }
}

class GetRelWordFeatureList extends AmbFunctionValue {
  override def apply(argumentValues: JList[Object], env: JEnv, b: ParametricBfgBuilder) = {
    Preconditions.checkArgument(argumentValues.size() == 2)
    val word = argumentValues.get(0).asInstanceOf[String]
    val featureComputer = argumentValues.get(1).asInstanceOf[SfeFeatureComputer]

    val features = featureComputer.getFeaturesForRelWord(word).asJava
    ConsValue.listToConsList(features)
  }
}

class GetEntityFeatures extends AmbFunctionValue {
  override def apply(argumentValues: JList[Object], env: JEnv, b: ParametricBfgBuilder) = {
    Preconditions.checkArgument(argumentValues.size() == 3)
    val entity = argumentValues.get(0).asInstanceOf[String]
    val word = argumentValues.get(1).asInstanceOf[String]
    val featureComputer = argumentValues.get(2).asInstanceOf[SfeFeatureComputer]
    featureComputer.getEntityFeatures(entity, word)
  }
}


class GetEntityFeatureDifference extends AmbFunctionValue {
  override def apply(argumentValues: JList[Object], env: JEnv, b: ParametricBfgBuilder) = {
    Preconditions.checkArgument(argumentValues.size() == 4)
    val entity = argumentValues.get(0).asInstanceOf[String]
    val neg_entity = argumentValues.get(1).asInstanceOf[String]
    val word = argumentValues.get(2).asInstanceOf[String]
    val featureComputer = argumentValues.get(3).asInstanceOf[SfeFeatureComputer]

    val positiveVector = featureComputer.getEntityFeatures(entity, word)
    val negativeVector = featureComputer.getEntityFeatures(neg_entity, word)
    positiveVector.elementwiseAddition(negativeVector.elementwiseProduct(-1))
  }
}

class GetEntityPairFeatures extends AmbFunctionValue {
  override def apply(argumentValues: JList[Object], env: JEnv, b: ParametricBfgBuilder) = {
    Preconditions.checkArgument(argumentValues.size() == 4)
    val entity1 = argumentValues.get(0).asInstanceOf[String]
    val entity2 = argumentValues.get(1).asInstanceOf[String]
    val word = argumentValues.get(2).asInstanceOf[String]
    val featureComputer = argumentValues.get(3).asInstanceOf[SfeFeatureComputer]

    featureComputer.getEntityPairFeatures(entity1, entity2, word)
  }
}


class GetEntityPairFeatureDifference extends AmbFunctionValue {
  override def apply(argumentValues: JList[Object], env: JEnv, b: ParametricBfgBuilder) = {
    Preconditions.checkArgument(argumentValues.size() == 6)
    val entity1 = argumentValues.get(0).asInstanceOf[String]
    val entity2 = argumentValues.get(1).asInstanceOf[String]
    val neg_entity1 = argumentValues.get(2).asInstanceOf[String]
    val neg_entity2 = argumentValues.get(3).asInstanceOf[String]
    val word = argumentValues.get(4).asInstanceOf[String]
    val featureComputer = argumentValues.get(5).asInstanceOf[SfeFeatureComputer]

    val positiveVector = featureComputer.getEntityPairFeatures(entity1, entity2, word)
    val negativeVector = featureComputer.getEntityPairFeatures(neg_entity1, neg_entity2, word)
    positiveVector.elementwiseAddition(negativeVector.elementwiseProduct(-1))
  }
}
