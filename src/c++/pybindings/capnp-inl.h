namespace fscpy {



namespace internal {
	template<>
	struct GetPipelineAsImpl<capnp::DynamicCapability> {
		static inline capnp::DynamicCapability::Client apply(DynamicValuePipeline& pipeline) {
			return pipeline.asCapability();
		}
	};
	
	template<>
	struct GetPipelineAsImpl<capnp::DynamicStruct> {
		static inline DynamicStructPipeline apply(DynamicValuePipeline& pipeline) {
			return pipeline.asStruct();
		}
	};
}

}