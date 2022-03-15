all:
	cd solvers && \
	  jbuilder build solver.exe && \
		jbuilder build clevrSolver.exe && \
	  jbuilder build versionDemo.exe && \
	  jbuilder build helmholtz.exe && \
	  jbuilder build logoDrawString.exe && \
	  jbuilder build protonet-tester.exe && \
	  jbuilder build compression.exe && \
	  cp _build/default/compression.exe ../bin/compression && \
	  cp _build/default/versionDemo.exe ../bin/versionDemo && \
	  cp _build/default/solver.exe ../bin/solver && \
		cp _build/default/clevrSolver.exe ../bin/clevrSolver && \
	  cp _build/default/helmholtz.exe ../bin/helmholtz && \
	  cp _build/default/protonet-tester.exe ../bin/protonet-tester && \
	  cp _build/default/logoDrawString.exe \
	    ../bin/logoDrawString
			
copy:
				cp ../ec_language/ec_language/solver .
				cp ../ec_language/ec_language/compression .
				cp ../ec_language/ec_language/helmholtz .
				cp ../ec_language/ec_language/logoDrawString .
				cp ../ec_language/ec_language/data/geom/logoDrawString data/geom/logoDrawString
clean:
	cd solvers && jbuilder clean
	rm -f bin/solver
	rm -f bin/clevrSolver
	rm -f bin/compression
	rm -f bin/helmholtz
	rm -f bin/logoDrawString

compressionClean:
	cd solvers && jbuilder clean
	rm -f bin/compression

logoDrawString:
	cd solvers && \
	jbuilder build logoDrawString.exe && \
	cp _build/default/logoDrawString.exe \
	    ../bin/logoDrawString

compression:
	cd solvers && \
	jbuilder build compression.exe && \
	cp _build/default/compression.exe ../bin/compression

compression_rescoring_api:
		cd solvers && jbuilder clean && \
		rm -f bin/compression_rescoring_api && \
		jbuilder build compression_rescoring_api.exe && \
		cp _build/default/compression_rescoring_api.exe ../bin/compression_rescoring_api

lcTestClean:
	cd solvers && jbuilder clean
	rm -f bin/lcTest

lcTest:
	cd solvers && \
	jbuilder build lcTest.exe && \
	cp _build/default/lcTest.exe ../bin/lcTest

clevrTestClean: 
	cd solvers && jbuilder clean
	rm -f bin/clevrTest

clevrTest:
	cd solvers && \
	jbuilder build clevrTest.exe && \
	cp _build/default/clevrTest.exe ../bin/clevrTest

solverClean: 
	cd solvers && jbuilder clean
	rm -f bin/solver

solver: 
	cd solvers && \
	jbuilder build solver.exe && \
	cp _build/default/solver.exe ../bin/solver

clevrClean: 
	cd solvers && jbuilder clean
	rm -f bin/clevrSolver

clevrSolver: 
	cd solvers && \
	jbuilder build clevrSolver.exe && \
	cp _build/default/clevrSolver.exe ../bin/clevrSolver

clevrPrimitivesTest:
	cd solvers && jbuilder clean 
	rm -f bin/test_clevr_primitives 
	cd solvers && jbuilder build test_clevr_primitives.exe && \
	cp _build/default/test_clevr_primitives.exe ../bin/test_clevr_primitives

re2TestClean: 
	cd solvers && jbuilder clean
	rm -f bin/re2Test
	
re2Test: 
		cd solvers && \
		jbuilder build re2Test.exe && \
		cp _build/default/re2Test.exe ../bin/re2Test

re2PrimsClean: 
	cd solvers && jbuilder clean
	rm -f bin/re2Primitives
	
re2Prims: 
		cd solvers && \
		jbuilder build re2Primitives.exe && \
		cp _build/default/re2Primitives.exe ../bin/re2Primitives