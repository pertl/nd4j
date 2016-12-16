package org.nd4j.linalg.api.ops.impl.meta;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.grid.OpDescriptor;

/**
 * This MetaOp covers case, when Op A and Op B are both using linear memory access
 *
 * You're NOT supposed to directly call this op. Do it on your own risk, only if you're absolutely have to.
 *
 * @author raver119@gmail.com
 */
public class PredicateMetaOp extends BaseMetaOp {

    public PredicateMetaOp() {

    }

    public PredicateMetaOp(INDArray x, INDArray y) {
        super(x, y);
    }

    public PredicateMetaOp(Op opA, Op opB) {
        super(opA, opB);
    }

    public PredicateMetaOp(OpDescriptor opA, OpDescriptor opB) {
        super(opA, opB);
    }

    public PredicateMetaOp(ScalarOp opA, TransformOp opB) {
        super(opA, opB);
    }

    public PredicateMetaOp(TransformOp opA, TransformOp opB) {
        super(opA, opB);
    }

    public PredicateMetaOp(TransformOp opA, ScalarOp opB) {
        super(opA, opB);
    }

    public PredicateMetaOp(ScalarOp opA, ScalarOp opB) {
        super(opA, opB);
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String name() {
        return "meta_predicate";
    }
}