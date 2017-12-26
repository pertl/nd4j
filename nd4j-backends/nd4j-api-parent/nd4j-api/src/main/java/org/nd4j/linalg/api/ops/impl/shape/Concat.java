package org.nd4j.linalg.api.ops.impl.shape;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Map;

@Slf4j
public class Concat extends DynamicCustomOp {
    private int concatDimension;

    @Override
    public String opName() {
        return "concat";
    }


    @Override
    public void resolvePropertiesFromSameDiffBeforeExecution() {
          val propertiesToResolve = sameDiff.propertiesToResolveForFunction(this);
          if(!propertiesToResolve.isEmpty()) {
              val varName = propertiesToResolve.get(0);
              val var = sameDiff.getVariable(varName);
              if(var == null) {
                  throw new ND4JIllegalStateException("No variable found with name " +varName);
              }
              else if(var.getArr() == null) {
                  throw new ND4JIllegalStateException("Array with variable name " + varName + " unset!");
              }

              concatDimension = var.getArr().getInt(0);
              addIArgument(concatDimension);
          }
    }

    @Override
    public void assertValidForExecution() {
        val descriptor = getDescriptor();
        if(descriptor == null)
            throw new NoOpNameFoundException("No descriptor found for op name " + opName());


        if(descriptor.getNumInputs() > 0 && numInputArguments() < 2)
            throw new ND4JIllegalStateException("Op failure for " + opName() + " Number of inputs is invalid for execution. Specified " + numInputArguments() + " but should be " + descriptor.getNumInputs());

        if(descriptor.getNumOutputs() > 0 && numOutputArguments() != descriptor.getNumOutputs())
            throw new ND4JIllegalStateException("Op failure for " + opName() + " Number of outputs is invalid for execution. Specified " + numOutputArguments() + " but should be " + descriptor.getNumInputs());

        //< 0 means dynamic size
        if(descriptor.getNumIArgs() >= 0 && numIArguments() != descriptor.getNumIArgs())
            throw new ND4JIllegalStateException("Op failure for " + opName() + " Number of integer arguments is invalid for execution. Specified " + numIArguments() + " but should be " + descriptor.getNumIArgs());

        if(descriptor.getNumTArgs() >= 0 && numTArguments() != descriptor.getNumTArgs())
            throw new ND4JIllegalStateException("Op failure for " + opName() + " Number of inputs is invalid for execution. Specified " + numTArguments() + " but should be " + descriptor.getNumTArgs());

    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        int concatDimension = -1;
        val input = nodeDef.getInput(nodeDef.getInputCount() - 1);
        val variable = initWith.getVariable(input);
        // concat dimension is only possible
        if (variable != null && variable.getArr() == null) {
            sameDiff.addPropertyToResolve(this, input);

        } else if (variable != null) {
            val arr = variable.getArr();
            if (arr.length() == 1) {
                concatDimension = arr.getInt(0);
            }

            // if that's convolution graph, we should swap dimensions
            if (concatDimension == 3)
                concatDimension = 1;

            this.concatDimension = concatDimension;
            addIArgument(this.concatDimension);
            log.debug("Concat dimension: {}", concatDimension);

        }
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }

    @Override
    public String onnxName() {
        return "Concat";
    }

    @Override
    public String tensorflowName() {
        return "Concat";
    }


    @Override
    public String[] tensorflowNames() {
        return new String[]  {"Concat","ConcatV2"};
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }
}
